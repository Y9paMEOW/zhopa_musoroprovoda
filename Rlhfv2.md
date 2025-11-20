Билиберт 🇸🇪:
Вот максимально простое применение RL для языковой модели:

1. БАЗОВЫЙ КОД (просто скопируй и запусти)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig

# 1. ЗАГРУЗКА МОДЕЛИ
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",  # или твоя модель
    device_map="auto",
    load_in_8bit=True  # экономия памяти
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. ПРОМПТЫ ДЛЯ ОБУЧЕНИЯ
prompts = [
    "Что такое искусственный интеллект?",
    "Объясни квантовую физику:",
    "Как работает блокчейн?",
    "Напиши рецепт пасты:"
]

# 3. ПРОСТАЯ REWARD-ФУНКЦИЯ (самое главное!)
def calculate_reward(texts):
    rewards = []
    for text in texts:
        reward = 0.0
        
        # Простые правила - настрой под свою задачу!
        if "интеллект" in text.lower():
            reward += 0.5
        if "алгоритм" in text.lower():
            reward += 0.3
        if len(text.split()) > 10:  # бонус за длинные ответы
            reward += 0.2
            
        rewards.append(reward)
    
    return rewards

# 4. НАСТРОЙКА PPO
config = PPOConfig(
    batch_size=2,
    learning_rate=1.41e-5,
)
ppo_trainer = PPOTrainer(config, model, tokenizer=tokenizer)

# 5. ОБУЧЕНИЕ (всего 3 шага для примера)
for step in range(3):
    print(f"=== Шаг {step + 1} ===")
    
    # Берем случайные промпты
    batch_prompts = [prompts[i % len(prompts)] for i in range(2)]
    
    # Токенизируем
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
    
    # Генерируем ответы
    response_tensors = ppo_trainer.generate(
        inputs["input_ids"],
        max_length=100,
        do_sample=True
    )
    
    # Декодируем
    responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
    
    # Вычисляем rewards
    rewards = calculate_reward(responses)
    
    # Шаг обучения
    stats = ppo_trainer.step(inputs["input_ids"], response_tensors, rewards)
    
    # Показываем что получилось
    for i, (prompt, response, reward) in enumerate(zip(batch_prompts, responses, rewards)):
        print(f"Prompt: {prompt}")
        print(f"Response: {response[:50]}...")
        print(f"Reward: {reward}")
        print("---")

print("Обучение завершено!")

2. КУДА ВСТАВИТЬ СВОИ ДАННЫЕ

Замени только эти части:

🔧 Промпты (строка 9):
# ТВОИ промпты:
prompts = [
    "Вопрос из твоей базы знаний 1",
    "Вопрос из твоей базы знаний 2", 
    "Вопрос из твоей базы знаний 3"
]

🔧 Reward-функция (строки 16-26):
def calculate_reward(texts):
    rewards = []
    for text in texts:
        reward = 0.0
        
        # ТВОИ правила оценки:
        if "правильный ответ" in text.lower():
            reward += 1.0
        if "спасибо" in text.lower():
            reward += 0.5
        if "точный" in text.lower():
            reward += 0.3
            
        # Штрафы:
        if "не знаю" in text.lower():
            reward -= 0.5
            
        rewards.append(reward)
    
    return rewards

🔧 Модель (строка 4):
model = AutoModelForCausalLM.from_pretrained(
    "твоя_модель",  # твоя модель
    device_map="auto", 
    load_in_8bit=True
)

3. ПРИМЕР ДЛЯ QA + RAG
# Простой пример для QA системы
def qa_reward_function(texts):
    rewards = []
    for text in texts:
        reward = 0.0
        
        # Бонусы за хорошие ответы:
        if "согласно контексту" in text.lower():
            reward += 0.5
        if "точный ответ" in text.lower():
            reward += 0.5
        if len(text) > 50:  # не слишком короткие
            reward += 0.3
            
        # Штрафы за плохие:
        if "не знаю" in text.lower():
            reward -= 0.7
        if "наверное" in text.lower():
            reward -= 0.2
            
        rewards.append(max(reward, 0.1))  # минимум 0.1
    
    return rewards

# Промпты для QA
qa_prompts = [
    "Пользователь: Что такое машинное обучение? Контекст: Машинное обучение - это...",
    "Пользователь: Как работает RAG? Контекст: Retrieval Augmented Generation...",

"Пользователь: Объясни трансформеры: Контекст: Трансформеры - архитектура NLP..."
]

4. КАК ЭТО РАБОТАЕТ (простыми словами):

1. Берем модель → Генерируем ответы на промпты
2. Оцениваем ответы → Reward-функция дает оценку (0.1, 0.5, 1.0)
3. Учим модель → Увеличиваем вероятность "хороших" ответов
4. Повторяем → Модель становится лучше

5. ЗАПУСК:
# Установи зависимости один раз:
pip install transformers trl torch peft

# Запусти код:
python rl_simple.py

6. ЧТО ПОМЕНЯТЬ ДЛЯ ТВОЕГО СЛУЧАЯ:

· Промпты → Возьми из своей базы вопросов
· Reward-функция → Опиши, что для тебя "хороший" ответ
· Модель → Укажи свою fine-tuned модель

Вот и всё! Этот код уже готов к работе. Просто подставь свои данные и запусти.
