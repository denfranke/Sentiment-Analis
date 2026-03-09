import requests
from bs4 import BeautifulSoup
import csv
import os


# Настройки
url = 'https://www.rustore.ru/catalog/app/ru.ozon.app.android/reviews/page-2'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.google.com/'
}
id=1

def scrape_reviews(url):
    global id

    # Отправляем GET-запрос
    print("Отправляю запрос к странице...")
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Ошибка загрузки страницы: {response.status_code}")
        return
    
    # Парсим HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Находим все блоки с отзывами
    # Пример: <div class="user-review"> или <div class="review-item"> — зависит от сайта
    review_blocks = soup.find_all('div', class_='Z8HHraBe JjR_4jS4 Q_8seuPA')  # ← измените селектор под нужный сайт
    
    if not review_blocks:
        print("Отзывы не найдены. Проверьте HTML-структуру страницы.")
        return
    
    reviews_data = []
    
    for idx, block in enumerate(review_blocks, 1):
        # Извлекаем текст отзыва
        # Например: <p class="review-text">Текст отзыва</p>
        review_text = block.find('p', class_='YOMjlbtO')
        text = review_text.get_text(strip=True) if review_text else "Текст не найден"
        
        # Извлекаем имя автора (если есть)
        author = block.find('span', class_='giIK213i')
        author_name = author.get_text(strip=True) if author else "Аноним"
        
        # Извлекаем рейтинг (если есть)
        rating = block.find('span', class_='EqjPUFIU YU7iNhGm')
        if rating:
            # Ищем все <use> внутри этого блока, у которых href заканчивается на #star_new_full
            full_stars = rating.find_all('use', attrs={'href': lambda x: x and '#star_new_full' in x})
            stars = len(full_stars)
        else:
            stars = "Без оценки"
        #stars = rating.find_all_next('use', href_='/catalog/_next/static/media/sprite.ff098203.svg#star_new_full').count.__str__ if rating else "Без оценки"
        
        # Сохраняем данные
        reviews_data.append({
            'Номер': id,
            'Автор': author_name,
            'Рейтинг': stars,
            'Текст': text
        })

        id=id+1
        
        #print(f"Отзыв {idx}: {text[:100]}...")  # Печать первых 100 символов
    
    # Сохранение в CSV
    save_to_csv(reviews_data)
    print(f"✅ Успешно собрано {len(reviews_data)} отзывов.")

def save_to_csv(data, filename="отзывы.csv"):
    file_exists = os.path.isfile(filename)
    write_header = not file_exists or os.path.getsize(filename) == 0

    with open(filename, 'a', encoding='utf-8', newline='', errors='replace') as f:
        fieldnames = ['Номер', 'Автор', 'Рейтинг', 'Текст']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Записываем заголовок только один раз
        if write_header:
            writer.writeheader()

        # Дозаписываем данные
        for row in data:
            writer.writerow(row)

    print("Данные сохранены в файл «отзывы.csv».")

def clear_csv_file(filename):
    """Очищает содержимое CSV-файла и записывает заголовок."""
    with open(filename, 'w', encoding='utf-8', newline='', errors='replace') as f:
        writer = csv.DictWriter(f, fieldnames=['Номер', 'Автор', 'Рейтинг', 'Текст'])
        writer.writeheader()

# Запуск
if __name__ == "__main__":
    clear_csv_file("отзывы.csv")

    total_pages = 1000
    target_url="https://www.rustore.ru/catalog/app/ru.ozon.app.android/reviews/page-"
    for page_num in range(1, total_pages+1):
        a=target_url+str(page_num)
        scrape_reviews(url=a)
