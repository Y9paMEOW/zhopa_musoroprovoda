Отличные вопросы! Давайте разберем оба момента подробно.

1. Что должно быть в sumocfg файле

Конфигурационный файл (.sumocfg) — это главный файл, который связывает все компоненты симуляции. Вот его типичная структура:

Базовый пример sumocfg файла:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    
    <!-- Входные файлы -->
    <input>
        <net-file value="network.net.xml"/>
        <route-files value="routes.rou.xml"/>
        <additional-files value="additional.add.xml"/>
    </input>
    
    <!-- Временные параметры -->
    <time>
        <begin value="0"/>
        <end value="3600"/> <!-- Симуляция на 1 час (3600 секунд) -->
    </time>
    
    <!-- Выходные данные -->
    <output>
        <tripinfo-output value="output/tripinfo.xml"/>
        <summary-output value="output/summary.xml"/>
        <queue-output value="output/queue.xml"/>
        <emission-output value="output/emission.xml"/>
        <fcd-output value="output/fcd.xml"/> <!-- Floating Car Data -->
    </output>
    
    <!-- Настройки обработки -->
    <processing>
        <ignore-route-errors value="true"/>
        <time-to-teleport value="300"/> <!-- Время до телепортации застрявших vehicles -->
    </processing>
    
    <!-- Отчеты -->
    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>

</configuration>
```

Расширенный пример с дополнительными опциями:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="my_network.net.xml"/>
        <route-files value="vehicles.rou.xml"/>
        <additional-files value="traffic_lights.add.xml, detectors.add.xml"/>
    </input>
    
    <time>
        <begin value="0"/>
        <end value="7200"/> <!-- 2 часа симуляции -->
        <step-length value="1"/> <!-- Длительность шага в секундах -->
    </time>
    
    <output>
        <!-- Основные выходные файлы -->
        <tripinfo-output value="results/tripinfo.xml"/>
        <summary-output value="results/summary.xml"/>
        <emission-output value="results/emissions.xml"/>
        
        <!-- Детализированные данные для анализа -->
        <fcd-output value="results/fcd.xml"/>
        <lanechange-output value="results/lanechange.xml"/>
        <vtk-output value="results/vtk/> <!-- Для визуализации в ParaView -->
        
        <!-- Статистика в человеко-читаемом формате -->
        <statistic-output value="results/statistics.xml"/>
    </output>
    
    <routing>
        <device.rerouting.probability value="0.2"/> <!-- Вероятность перемаршрутизации -->
    </routing>
    
    <random_number>
        <seed value="12345"/> <!-- Seed для воспроизводимости -->
    </random_number>
</configuration>
```

2. Запуск симуляций и сохранение результатов для обработки в Python

Да, абсолютно! Это стандартный и рекомендуемый workflow. Вот как это делается:

Полный код для запуска и анализа:

```python
import os
import subprocess
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class SUMOAnalyzer:
    def __init__(self, config_path, output_dir="results"):
        self.config_path = config_path
        self.output_dir = output_dir
        self.results = {}
        
        # Создаем директорию для результатов
        os.makedirs(output_dir, exist_ok=True)
    
    def run_simulation(self, use_gui=False):
        """Запуск симуляции и сохранение результатов"""
        print(f"Запуск симуляции из конфигурации: {self.config_path}")
        
        try:
            if use_gui:
                # Для визуальной проверки
                subprocess.run(['sumo-gui', '-c', self.config_path], check=True)
            else:
                # Для быстрого выполнения и сбора данных
                subprocess.run(['sumo', '-c', self.config_path], check=True)
            
            print("Симуляция успешно завершена!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при запуске симуляции: {e}")
            return False
    
    def parse_tripinfo(self, tripinfo_file):
        """Парсинг информации о поездках"""
        print(f"Анализ файла поездок: {tripinfo_file}")
        
        try:
            tree = ET.parse(tripinfo_file)
            root = tree.getroot()
            
            trips_data = []
            for trip in root.findall('tripinfo'):
                trip_data = {
                    'id': trip.get('id'),
                    'depart': float(trip.get('depart', 0)),
                    'arrival': float(trip.get('arrival', 0)),
                    'duration': float(trip.get('duration', 0)),
                    'routeLength': float(trip.get('routeLength', 0)),
                    'waitingTime': float(trip.get('waitingTime', 0)),
                    'timeLoss': float(trip.get('timeLoss', 0)),
                    'departLane': trip.get('departLane', ''),
                    'arrivalLane': trip.get('arrivalLane', ''),
                    'departPos': float(trip.get('departPos', 0)),
                    'arrivalPos': float(trip.get('arrivalPos', 0))
                }
                trips_data.append(trip_data)
            
            df = pd.DataFrame(trips_data)
            self.results['tripinfo'] = df
            return df
            
        except Exception as e:
            print(f"Ошибка при чтении tripinfo: {e}")
            return pd.DataFrame()
    
    def parse_emissions(self, emission_file):
        """Парсинг данных о выбросах"""
        print(f"Анализ файла выбросов: {emission_file}")
        
        try:
            tree = ET.parse(emission_file)
            root = tree.getroot()
            
            emissions_data = []
            for interval in root.findall('interval'):
                emission_data = {
                    'begin': float(interval.get('begin', 0)),
                    'end': float(interval.get('end', 0)),
                    'CO2_abs': float(interval.get('CO2_abs', 0)),
                    'CO_abs': float(interval.get('CO_abs', 0)),
                    'HC_abs': float(interval.get('HC_abs', 0)),
                    'NOx_abs': float(interval.get('NOx_abs', 0)),
                    'fuel_abs': float(interval.get('fuel_abs', 0))
                }
                emissions_data.append(emission_data)
            
            df = pd.DataFrame(emissions_data)
            self.results['emissions'] = df
            return df
            
        except Exception as e:
            print(f"Ошибка при чтении emission data: {e}")
            return pd.DataFrame()
    
    def parse_summary(self, summary_file):
        """Парсинг суммарной статистики"""
        print(f"Анализ файла статистики: {summary_file}")
        
        try:
            tree = ET.parse(summary_file)
            root = tree.getroot()
            
            summary_data = {}
            
            # Основная статистика
            for step in root.findall('step'):
                summary_data['time'] = float(step.get('time', 0))
                summary_data['loaded_vehicles'] = int(step.get('loaded', 0))
                summary_data['running_vehicles'] = int(step.get('running', 0))
                summary_data['waiting_vehicles'] = int(step.get('waiting', 0))
                summary_data['ended_vehicles'] = int(step.get('ended', 0))
            
            self.results['summary'] = summary_data
            return summary_data
            
        except Exception as e:
            print(f"Ошибка при чтении summary: {e}")
            return {}
    
    def generate_comprehensive_report(self):
        """Генерация комплексного отчета"""
        if not self.results:
            print("Нет данных для отчета")
            return
        
        print("\n" + "="*60)
        print("КОМПЛЕКСНЫЙ ОТЧЕТ ПО СИМУЛЯЦИИ")
        print("="*60)
        
        # Анализ поездок
        if 'tripinfo' in self.results:
            df_trips = self.results['tripinfo']
            print(f"\n--- СТАТИСТИКА ПОЕЗДОК ---")
            print(f"Всего поездок: {len(df_trips)}")
            print(f"Среднее время поездки: {df_trips['duration'].mean():.2f} сек")
            print(f"Средняя скорость: {(df_trips['routeLength'] / df_trips['duration']).mean():.2f} м/с")
            print(f"Общее время ожидания: {df_trips['waitingTime'].sum():.2f} сек")
            print(f"Максимальное время поездки: {df_trips['duration'].max():.2f} сек")
        
        # Анализ выбросов
        if 'emissions' in self.results:
            df_emissions = self.results['emissions']
            print(f"\n--- ЭКОЛОГИЧЕСКИЕ ПОКАЗАТЕЛИ ---")
            print(f"Общие выбросы CO2: {df_emissions['CO2_abs'].sum():.2f} мг")
            print(f"Общие выбросы CO: {df_emissions['CO_abs'].sum():.2f} мг")
            print(f"Общий расход топлива: {df_emissions['fuel_abs'].sum():.2f} мл")
    
    def create_visualizations(self):
        """Создание визуализаций результатов"""
        if not self.results:
            print("Нет данных для визуализации")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Анализ результатов SUMO симуляции', fontsize=16)
        
        # Распределение времени поездки
        if 'tripinfo' in self.results:
            df_trips = self.results['tripinfo']
            
            axes[0,0].hist(df_trips['duration'], bins=30, alpha=0.7, color='skyblue')
            axes[0,0].set_title('Распределение времени поездки')
            axes[0,0].set_xlabel('Время (сек)')
            axes[0,0].set_ylabel('Количество поездок')
            
            # Длина маршрута vs Время поездки
            axes[0,1].scatter(df_trips['routeLength'], df_trips['duration'], alpha=0.5)
            axes[0,1].set_title('Длина маршрута vs Время поездки')
            axes[0,1].set_xlabel('Длина маршрута (м)')
            axes[0,1].set_ylabel('Время поездки (сек)')
            
            # Время ожидания
            axes[0,2].hist(df_trips['waitingTime'], bins=30, alpha=0.7, color='lightcoral')
            axes[0,2].set_title('Распределение времени ожидания')
            axes[0,2].set_xlabel('Время ожидания (сек)')
            axes[0,2].set_ylabel('Количество поездок')
        
        # Анализ выбросов во времени
        if 'emissions' in self.results:
            df_emissions = self.results['emissions']
            
            axes[1,0].plot(df_emissions['begin'], df_emissions['CO2_abs'])
            axes[1,0].set_title('Выбросы CO2 по времени')
            axes[1,0].set_xlabel('Время (сек)')
            axes[1,0].set_ylabel('CO2 (мг)')
            
            axes[1,1].plot(df_emissions['begin'], df_emissions['fuel_abs'])
            axes[1,1].set_title('Расход топлива по времени')
            axes[1,1].set_xlabel('Время (сек)')
            axes[1,1].set_ylabel('Топливо (мл)')
            
            # Корреляционная матрица (если есть достаточно данных)
            if len(df_emissions) > 2:
                numeric_cols = ['CO2_abs', 'CO_abs', 'HC_abs', 'NOx_abs', 'fuel_abs']
                correlation = df_emissions[numeric_cols].corr()
                im = axes[1,2].imshow(correlation, cmap='coolwarm', aspect='auto')
                axes[1,2].set_title('Корреляция выбросов')
                axes[1,2].set_xticks(range(len(numeric_cols)))
                axes[1,2].set_yticks(range(len(numeric_cols)))
                axes[1,2].set_xticklabels(numeric_cols, rotation=45)
                axes[1,2].set_yticklabels(numeric_cols)
                plt.colorbar(im, ax=axes[1,2])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results_to_csv(self):
        """Сохранение результатов в CSV для дальнейшего анализа"""
        for data_type, data in self.results.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                filename = f"{self.output_dir}/{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                data.to_csv(filename, index=False)
                print(f"Данные сохранены: {filename}")
    
    def run_complete_analysis(self, use_gui=False):
        """Полный цикл анализа"""
        # Запуск симуляции
        success = self.run_simulation(use_gui=use_gui)
        
        if success:
            # Анализ результатов (предполагаем стандартные имена файлов)
            self.parse_tripinfo("output/tripinfo.xml")
            self.parse_emissions("output/emission.xml")
            self.parse_summary("output/summary.xml")
            
            # Генерация отчетов
            self.generate_comprehensive_report()
            self.create_visualizations()
            self.save_results_to_csv()
            
            return True
        return False

# Пример использования
if __name__ == "__main__":
    # Укажите путь к вашему конфигурационному файлу
    config_file = "your_config.sumocfg"
    
    # Создаем анализатор
    analyzer = SUMOAnalyzer(config_file, output_dir="simulation_results")
    
    # Запускаем полный анализ (use_gui=False для быстрого выполнения)
    analyzer.run_complete_analysis(use_gui=False)
```

Дополнительный код для пакетного запуска симуляций:

```python
import glob

def batch_simulations(config_pattern="scenarios/*.sumocfg"):
    """Запуск нескольких симуляций подряд"""
    config_files = glob.glob(config_pattern)
    
    results = {}
    for config_file in config_files:
        print(f"\nЗапуск симуляции: {config_file}")
        scenario_name = os.path.basename(config_file).replace('.sumocfg', '')
        
        analyzer = SUMOAnalyzer(config_file, output_dir=f"results/{scenario_name}")
        success = analyzer.run_complete_analysis(use_gui=False)
        
        if success and 'tripinfo' in analyzer.results:
            # Сохраняем ключевые метрики
            df = analyzer.results['tripinfo']
            results[scenario_name] = {
                'avg_travel_time': df['duration'].mean(),
                'total_trips': len(df),
                'total_waiting_time': df['waitingTime'].sum()
            }
    
    # Сравнительный анализ
    comparison_df = pd.DataFrame(results).T
    print("\nСравнительный анализ сценариев:")
    print(comparison_df)
    
    return comparison_df
```

Ключевые преимущества этого подхода:

1. Автоматизация — Запуск множества симуляций без ручного вмешательства
2. Воспроизводимость — Все параметры сохраняются в конфигурационных файлах
3. Глубокий анализ — Полный доступ ко всем данным симуляции
4. Гибкость — Легко модифицировать под конкретные задачи
5. Масштабируемость — Можно анализировать результаты множества экспериментов

Этот workflow позволяет вам сосредоточиться на анализе результатов, а не на рутинных операциях запуска симуляций!
