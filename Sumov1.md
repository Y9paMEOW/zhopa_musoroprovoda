Отлично! Если у вас уже есть готовая конфигурация, процесс значительно упрощается. Вот пошаговый код и инструкция для начала работы.

Базовый рабочий процесс с готовой конфигурацией

Структура типичного проекта SUMO

Предположим, у вас есть:

```
project_folder/
├── your_config.sumocfg
├── network.net.xml
├── routes.rou.xml
├── additional.add.xml (опционально - светофоры, детекторы)
└── output/ (папка для результатов)
```

1. Базовый запуск симуляции

Вариант A: Графический интерфейс (рекомендуется для начала)

```bash
# В командной строке перейдите в папку проекта и выполните:
sumo-gui -c your_config.sumocfg
```

Вариант B: Командная строка (для быстрых запусков)

```bash
sumo -c your_config.sumocfg
```

2. Python скрипт для автоматизации и анализа

Создайте файл run_simulation.py:

```python
import os
import subprocess
import sys
import sumolib
import traci
from collections import defaultdict
import matplotlib.pyplot as plt

class SUMOSimulation:
    def __init__(self, config_file):
        self.config_file = config_file
        self.simulation_data = defaultdict(list)
        
    def run_basic_simulation(self):
        """Запуск симуляции через командную строку"""
        try:
            print("Запуск SUMO симуляции...")
            # Для графического интерфейса используйте 'sumo-gui'
            result = subprocess.run(['sumo', '-c', self.config_file], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("Симуляция успешно завершена!")
                return True
            else:
                print(f"Ошибка при запуске: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("Ошибка: SUMO не найден. Убедитесь, что SUMO установлен и добавлен в PATH")
            return False
    
    def run_with_traci(self, simulation_time=1000):
        """Запуск симуляции с использованием TraCI для сбора данных"""
        try:
            # Запускаем SUMO как сервер
            sumo_binary = "sumo"
            config_path = os.path.abspath(self.config_file)
            
            traci.start([sumo_binary, "-c", config_path])
            
            print("Симуляция запущена с TraCI...")
            
            # Основной цикл симуляции
            step = 0
            while step < simulation_time:
                traci.simulationStep()
                
                # Сбор базовой статистики на каждом шаге
                self.collect_simulation_data(step)
                
                step += 1
                
                # Прогресс
                if step % 100 == 0:
                    print(f"Выполнен шаг {step}/{simulation_time}")
            
            # Завершение
            traci.close()
            print("Симуляция завершена")
            return True
            
        except Exception as e:
            print(f"Ошибка при работе с TraCI: {e}")
            return False
    
    def collect_simulation_data(self, step):
        """Сбор данных во время симуляции"""
        try:
            # Количество транспортных средств в симуляции
            vehicle_count = traci.vehicle.getIDCount()
            
            # Средняя скорость
            speeds = []
            for veh_id in traci.vehicle.getIDList():
                speed = traci.vehicle.getSpeed(veh_id)
                speeds.append(speed)
            
            avg_speed = sum(speeds) / len(speeds) if speeds else 0
            
            # Сохраняем данные
            self.simulation_data['step'].append(step)
            self.simulation_data['vehicle_count'].append(vehicle_count)
            self.simulation_data['avg_speed'].append(avg_speed)
            
        except Exception as e:
            print(f"Ошибка при сборе данных: {e}")
    
    def generate_basic_report(self):
        """Генерация базового отчета"""
        if not self.simulation_data:
            print("Нет данных для отчета")
            return
        
        print("\n" + "="*50)
        print("БАЗОВЫЙ ОТЧЕТ ПО СИМУЛЯЦИИ")
        print("="*50)
        
        max_vehicles = max(self.simulation_data['vehicle_count'])
        avg_speed_overall = sum(self.simulation_data['avg_speed']) / len(self.simulation_data['avg_speed'])
        
        print(f"Максимальное количество транспортных средств: {max_vehicles}")
        print(f"Средняя скорость за всю симуляцию: {avg_speed_overall:.2f} м/с")
        print(f"Общее время симуляции: {len(self.simulation_data['step'])} шагов")
    
    def plot_basic_metrics(self):
        """Построение простых графиков"""
        if not self.simulation_data:
            print("Нет данных для визуализации")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # График количества транспортных средств
        ax1.plot(self.simulation_data['step'], self.simulation_data['vehicle_count'])
        ax1.set_title('Количество транспортных средств по времени')
        ax1.set_ylabel('Количество')
        ax1.grid(True)
        
        # График средней скорости
        ax2.plot(self.simulation_data['step'], self.simulation_data['avg_speed'])
        ax2.set_title('Средняя скорость по времени')
        ax2.set_xlabel('Шаг симуляции')
        ax2.set_ylabel('Скорость (м/с)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('simulation_results.png', dpi=300)
        plt.show()

def main():
    # Укажите путь к вашему конфигурационному файлу
    config_file = "your_config.sumocfg"  # Замените на ваш файл
    
    # Проверяем существование файла
    if not os.path.exists(config_file):
        print(f"Ошибка: Файл конфигурации '{config_file}' не найден!")
        print("Убедитесь, что файл существует в текущей директории")
        return
    
    # Создаем экземпляр симуляции
    sim = SUMOSimulation(config_file)
    
    # Запускаем базовую симуляцию
    print("1. Запуск базовой симуляции...")
    success = sim.run_basic_simulation()
    
    if success:
        # Запускаем расширенную симуляцию с сбором данных
        print("\n2. Запуск симуляции со сбором данных...")
        sim.run_with_traci(simulation_time=1000)  # 1000 шагов
        
        # Генерируем отчет
        print("\n3. Генерация отчета...")
        sim.generate_basic_report()
        
        # Строим графики
        print("\n4. Построение графиков...")
        sim.plot_basic_metrics()
        
        print("\nРабота завершена! Проверьте файл 'simulation_results.png'")

if __name__ == "__main__":
    main()
```

3. Дополнительный скрипт для анализа выходных данных

Создайте analyze_output.py для анализа результатов:

```python
import sumolib
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def analyze_sumo_output():
    """Анализ выходных данных SUMO"""
    
    # Чтение файла статистики (если он настроен в конфиге)
    try:
        # Пример чтения XML файла с данными
        tree = ET.parse('output/tripinfo.xml')  # Укажите ваш файл
        root = tree.getroot()
        
        data = []
        for trip in root.findall('tripinfo'):
            trip_data = {
                'id': trip.get('id'),
                'depart': float(trip.get('depart', 0)),
                'arrival': float(trip.get('arrival', 0)),
                'duration': float(trip.get('duration', 0)),
                'routeLength': float(trip.get('routeLength', 0)),
                'waitingTime': float(trip.get('waitingTime', 0))
            }
            data.append(trip_data)
        
        df = pd.DataFrame(data)
        
        # Базовая статистика
        print("=== СТАТИСТИКА ПОЕЗДОК ===")
        print(f"Всего поездок: {len(df)}")
        print(f"Среднее время поездки: {df['duration'].mean():.2f} сек")
        print(f"Средняя длина маршрута: {df['routeLength'].mean():.2f} м")
        print(f"Общее время ожидания: {df['waitingTime'].sum():.2f} сек")
        
        # Визуализация
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        df['duration'].hist(ax=axes[0,0], bins=20)
        axes[0,0].set_title('Распределение времени поездки')
        axes[0,0].set_xlabel('Время (сек)')
        
        df['routeLength'].hist(ax=axes[0,1], bins=20)
        axes[0,1].set_title('Распределение длины маршрута')
        axes[0,1].set_xlabel('Длина (м)')
        
        df.plot.scatter(x='routeLength', y='duration', ax=axes[1,0])
        axes[1,0].set_title('Длина маршрута vs Время поездки')
        axes[1,0].set_xlabel('Длина (м)')
        axes[1,0].set_ylabel('Время (сек)')
        
        df['waitingTime'].hist(ax=axes[1,1], bins=20)
        axes[1,1].set_title('Время ожидания')
        axes[1,1].set_xlabel('Время (сек)')
        
        plt.tight_layout()
        plt.savefig('trip_analysis.png', dpi=300)
        plt.show()
        
    except FileNotFoundError:
        print("Выходные файлы не найдены. Убедитесь, что в конфигурации включен вывод tripinfo.")

if __name__ == "__main__":
    analyze_sumo_output()
```

4. Требования и установка зависимостей

Создайте requirements.txt:

```txt
sumolib
traci
matplotlib
pandas
numpy
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

Пошаговая инструкция:

1. Подготовка: Убедитесь, что SUMO установлен и добавлен в PATH
2. Размещение файлов: Поместите вашу конфигурацию и связанные файлы в папку проекта
3. Запуск: Выполните python run_simulation.py
4. Анализ: Запустите python analyze_output.py для детального анализа

Что вы получите:

· ✅ Визуализацию симуляции в SUMO-GUI
· ✅ Базовую статистику по транспортным потокам
· ✅ Графики изменения параметров во времени
· ✅ Анализ эффективности транспортной системы

Этот код даст вам хорошую стартовую точку для работы с любой конфигурацией SUMO!
