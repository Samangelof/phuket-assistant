import logging
import asyncio
import datetime
from typing import Dict, Tuple, Optional, Generator, Any
import httpx
import openai
from plugin_manager import PluginManager


class AssistantHelper:
    def __init__(self, config: dict, plugin_manager=PluginManager):
        """
        Инициализация помощника для работы с OpenAI Assistant API.
        
        Args:
            config: Словарь конфигурации с ключами api_key, assistant_id и т.д.
            plugin_manager: Опциональный менеджер плагинов
        """
        http_client = httpx.AsyncClient(
            proxy=config['proxy']) if 'proxy' in config else None
        self.client = openai.AsyncOpenAI(
            api_key=config['api_key'], http_client=http_client)
        self.config = config
        self.plugin_manager = plugin_manager

        self.assistant_id = config['assistant_id']
        # Хранение связи между идентификаторами чатов и потоками ассистента
        self.assistant_threads: Dict[int, str] = {}
        # Отслеживание времени последнего обновления для каждого чата
        self.last_updated: Dict[int, datetime.datetime] = {}

    async def get_chat_response(self, chat_id: int, query: str) -> Tuple[str, str]:
        """
        Получение ответа от ассистента.
        
        Args:
            chat_id: Идентификатор чата
            query: Текст запроса пользователя
            
        Returns:
            Tuple[str, str]: (Ответ ассистента, количество токенов)
        """
        try:
            current_date = datetime.datetime.now().strftime('%d-%m-%Y')
            query_with_date = f"Сегодня {current_date}. {query}"
            logging.info(f'[get_chat_response] {query_with_date}')
            
            thread_id = await self._get_or_create_thread(chat_id)
            
            # Создание сообщения пользователя
            await self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=query_with_date
            )
            
            tools = []
            if self.plugin_manager:
                function_specs = self.plugin_manager.get_functions_specs()
                if function_specs:
                    tools = [{"type": "function", "function": spec} for spec in function_specs]
                    
            # Запуск выполнения запроса
            run = await self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id,
                tools=tools
            )
            
            # Ожидание завершения выполнения
            run_status = await self._wait_for_run_completion(thread_id, run.id)
            
            if run_status.status == "completed":
                messages = await self.client.beta.threads.messages.list(
                    thread_id=thread_id,
                    order="desc",
                    limit=1
                )
                
                if messages.data and messages.data[0].role == "assistant" and messages.data[0].content:
                    answer = messages.data[0].content[0].text.value
                    estimated_tokens = len(query) + len(answer) // 4
                    
                    # Добавляем информацию об использовании, если это требуется
                    # if self.config.get('show_usage', False):
                    #     answer += f"\n\n---\n💰 {estimated_tokens} токенов"
                    
                    return answer, str(estimated_tokens)
                else:
                    return "Не удалось получить ответ от ассистента.", "0"
            else:
                return f"Ошибка в работе ассистента. Статус: {run_status.status}", "0"
        except Exception as e:
            logging.error(f"Ошибка при обращении к ассистенту: {e}")
            return f"⚠️ Произошла ошибка. ⚠️\n{str(e)}", "0"

    async def get_chat_response_stream(self, chat_id: int, query: str) -> Generator[Tuple[str, str], None, None]:
        """
        Получение ответа от ассистента в режиме потока с поддержкой плагинов.
        """
        try:
            current_date = datetime.datetime.now().strftime('%d-%m-%Y')
            query_with_date = f"Сегодня {current_date}. {query}"
            logging.info(f'[get_chat_response_stream] {query_with_date}')
            thread_id = await self._get_or_create_thread(chat_id)
            
            # Создание сообщения пользователя
            await self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=query_with_date
            )
            
            # Получаем спецификации функций от плагинов
            tools = []
            if self.plugin_manager:
                function_specs = self.plugin_manager.get_functions_specs()
                if function_specs:
                    tools = [{"type": "function", "function": spec} for spec in function_specs]
            
            # Запуск выполнения запроса с инструментами
            run = await self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id,
                tools=tools
            )
            
            full_answer = ""
            is_complete = False
            
            while not is_complete:
                run_status = await self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )

                logging.info(f"Статус выполнения: {run_status.status}")
                
                if run_status.status == "requires_action":
                    # Обработка вызовов функций
                    tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                    tool_outputs = []
                    
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        arguments = tool_call.function.arguments
                        
                        # Вызвать функцию через менеджер плагинов
                        result = await self.plugin_manager.call_function(
                            function_name=function_name,
                            helper=self,
                            arguments=arguments
                        )
                        
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": result
                        })
                    
                    # Отправить результаты обратно
                    await self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
                    
                    # Вывод начальной информации о том, что идет обработка
                    if not full_answer:
                        yield "Получаю данные из внешних источников...", "not_finished"
                
                elif run_status.status == "completed":
                    messages = await self.client.beta.threads.messages.list(
                        thread_id=thread_id,
                        order="desc",
                        limit=1
                    )
                    
                    if messages.data and messages.data[0].role == "assistant" and messages.data[0].content:
                        full_answer = messages.data[0].content[0].text.value
                        is_complete = True
                        estimated_tokens = len(query) + len(full_answer) // 4
                        
                        yield full_answer, str(estimated_tokens)
                    else:
                        yield "Не удалось получить ответ от ассистента.", "0"
                        is_complete = True
                
                elif run_status.status in ["failed", "cancelled", "expired"]:
                    yield f"Ошибка в работе ассистента. Статус: {run_status.status}", "0"
                    is_complete = True
                
                else:
                    if not full_answer:
                        yield "Генерирую ответ...", "not_finished"
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logging.error(f"Ошибка при обращении к ассистенту: {e}", exc_info=True)  # Добавлен вывод полного traceback
            yield f"⚠️ Произошла ошибка. ⚠️\n{str(e)}", "0"

    async def _get_or_create_thread(self, chat_id: int) -> str:
        """
        Получение существующего или создание нового потока ассистента.
        
        Args:
            chat_id: Идентификатор чата
            
        Returns:
            str: Идентификатор потока ассистента
        """
        if chat_id not in self.assistant_threads or self._max_age_reached(chat_id):
            thread = await self.client.beta.threads.create()
            self.assistant_threads[chat_id] = thread.id
            
        self.last_updated[chat_id] = datetime.datetime.now()
        return self.assistant_threads[chat_id]

    async def _wait_for_run_completion(self, thread_id: str, run_id: str, timeout: int = 120) -> Any:
        """
        Ожидание завершения выполнения запроса.
        
        Args:
            thread_id: Идентификатор потока
            run_id: Идентификатор запуска
            timeout: Максимальное время ожидания в секундах
            
        Returns:
            Статус выполнения запроса
            
        Raises:
            TimeoutError: Если превышено максимальное время ожидания
        """
        start_time = datetime.datetime.now()
        
        while True:
            run_status = await self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            if run_status.status in ["completed", "failed", "cancelled", "expired"]:
                return run_status
                
            if (datetime.datetime.now() - start_time).seconds > timeout:
                await self.client.beta.threads.runs.cancel(
                    thread_id=thread_id,
                    run_id=run_id
                )
                raise TimeoutError(f"Run {run_id} timed out after {timeout} seconds")
                
            await asyncio.sleep(1)

    async def _wait_for_run_completion_with_tools(self, thread_id: str, run_id: str):
        """Ждет завершения запуска, обрабатывая вызовы инструментов по мере необходимости."""
        while True:
            run_status = await self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            if run_status.status == "requires_action":
                # Обработка вызовов функций
                tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    arguments = tool_call.function.arguments
                    
                    # Вызвать функцию через менеджер плагинов
                    result = await self.plugin_manager.call_function(
                        function_name=function_name,
                        helper=self,
                        arguments=arguments
                    )
                    
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": result
                    })
                
                # Отправить результаты обратно
                await self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run_id,
                    tool_outputs=tool_outputs
                )
            
            elif run_status.status in ["completed", "failed", "cancelled", "expired"]:
                return run_status
            
            await asyncio.sleep(1)

    def reset_chat_history(self, chat_id: int) -> None:
        """
        Сброс истории чата.
        
        Args:
            chat_id: Идентификатор чата
        """
        if chat_id in self.assistant_threads:
            del self.assistant_threads[chat_id]
            if chat_id in self.last_updated:
                del self.last_updated[chat_id]

    def _max_age_reached(self, chat_id: int) -> bool:
        """
        Проверка, превышен ли максимальный возраст чата.
        
        Args:
            chat_id: Идентификатор чата
            
        Returns:
            bool: True, если максимальный возраст превышен, иначе False
        """
        if chat_id not in self.last_updated:
            return False
            
        last_updated = self.last_updated[chat_id]
        now = datetime.datetime.now()
        max_age_minutes = self.config.get('max_conversation_age_minutes', 60)
        
        return last_updated < now - datetime.timedelta(minutes=max_age_minutes)

    def get_conversation_stats(self, chat_id: int) -> Tuple[int, int]:
        """
        Получение статистики разговора.
        
        Args:
            chat_id: Идентификатор чата
            
        Returns:
            Tuple[int, int]: (количество сообщений, количество токенов)
        """
        # Этот метод должен быть реализован для Assistant API
        return 0, 0
        
    def get_prompt(self) -> str:
        """
        Получение текущего промпта.
        
        Returns:
            str: Информационное сообщение о промпте
        """
        return "Промпт управляется в интерфейсе Assistants API"
        
    def set_prompt(self, new_prompt: str) -> None:
        """
        Установка нового промпта.
        
        Args:
            new_prompt: Новый промпт
        """
        logging.info("Установка промпта через код не поддерживается для Assistant API")