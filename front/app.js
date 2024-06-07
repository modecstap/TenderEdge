document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const tableBody = document.querySelector('#dataTable tbody');

    form.addEventListener('submit', function(event) {
        event.preventDefault(); // Предотвращаем стандартное поведение формы

        const formData = new FormData(form);

        const params = {
            "vender": formData.get('vendor'),
            "functional": formData.get('functionality'),
            "price": formData.get('device-cost'),
            "refueling_cost": formData.get('consumables-cost'),
            "supplie_cost": formData.get('supplie_cost'),
            "repairability": formData.get('repairability'),
            "parts_support": formData.get('parts-support'),
            "manufacturer": formData.get('manufacturer'),
            "efficiency": formData.get('efficiency')
        };

        // Создаём строку запроса
        const queryString = new URLSearchParams(params.data).toString();

        console.log(queryString)

        // Отправляем POST-запрос
        fetch('http://localhost:8080/getCluster', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        })
        .then(response => {
            console.log('Ответ сервера:', response);
            if (!response.ok) {
                throw new Error('Network response was not ok ' + response.statusText);
            }
            return response.json();
        })
        .then(dataArray => {
            // Очищаем таблицу перед добавлением новых данных
            tableBody.innerHTML = '';

            // Обрабатываем массив данных, полученных от сервера
            dataArray.forEach(data => {
                // Создаём новую строку таблицы с полученными данными
                const newRow = document.createElement('tr');
                newRow.innerHTML = `
                    <td>${data.name}</td>
                    <td>${data.vender}</td>
                    <td>${data.functional}</td>
                    <td>${data.price}</td>
                    <td>${data.refueling_cost}</td>
                    <td>${data.supplie_cost}</td>
                    <td>${data.repairability}</td>
                    <td>${data.parts_support}</td>
                    <td>${data.manufacturer}</td>
                    <td>${data.efficiency}</td>
                `;

                // Добавляем новую строку в таблицу
                tableBody.appendChild(newRow);
            });
        })
        .catch(error => console.error('Ошибка:', error));
    });
});