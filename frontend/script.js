document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        handleFiles(files);
    });

    fileInput.addEventListener('change', () => {
        const files = fileInput.files;
        handleFiles(files);
    });

    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    function handleFiles(files) {
        for (const file of files) {
            uploadFile(file);
            const listItem = document.createElement('li');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.name = 'file';
            checkbox.value = file.name;
            listItem.appendChild(checkbox);
            listItem.appendChild(document.createTextNode(file.name));
            fileList.appendChild(listItem);
        }
    }

    function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/', true);

        xhr.onload = function () {
            if (xhr.status === 200) {
                console.log('File uploaded successfully');
            } else {
                console.error('File upload failed');
            }
        };

        xhr.send(formData);
    }

    function sendMessage() {
        const message = messageInput.value.trim();
        if (message === '') return;

        appendMessage('user', message);
        messageInput.value = '';

        const selectedFiles = Array.from(document.querySelectorAll('#file-list input[type="checkbox"]:checked'))
            .map(checkbox => checkbox.value);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/chat', true);
        xhr.setRequestHeader('Content-Type', 'application/json');

        xhr.onload = function () {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                appendMessage('bot', response.message);
            } else {
                console.error('Chat request failed');
            }
        };

        xhr.send(JSON.stringify({ message, files: selectedFiles }));
    }

    function appendMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        messageElement.innerText = message;
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
