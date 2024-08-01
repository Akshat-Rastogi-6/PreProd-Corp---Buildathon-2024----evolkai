function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(function(section) {
        section.style.display = 'none';
    });

    // Show the selected section
    document.getElementById(sectionId).style.display = 'block';
}

// Optionally, show the first section by default
document.addEventListener('DOMContentLoaded', function() {
    showSection('data-ingestion');
});


// uploadig data
document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();
    
    var formData = new FormData(this);
    
    fetch('upload.php', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            document.getElementById('file-name').textContent = 'File Name: ' + data.file_name;
            document.getElementById('file-dimensions').textContent = 'Dimensions: ' + data.rows + ' rows, ' + data.columns + ' columns';
        } else {
            document.getElementById('file-name').textContent = '';
            document.getElementById('file-dimensions').textContent = 'Error: ' + data.message;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('file-dimensions').textContent = 'Error: ' + error;
    });
});
document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();
    
    var formData = new FormData(this);
    
    fetch('upload.php', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(data => {
        // Split response by '|'
        const [status, fileName, rows, columns] = data.split('|');
        
        if (status === 'success') {
            document.getElementById('file-name').textContent = 'File Name: ' + fileName;
            document.getElementById('file-dimensions').textContent = 'Dimensions: ' + rows + ' rows, ' + columns + ' columns';
        } else {
            document.getElementById('file-name').textContent = '';
            document.getElementById('file-dimensions').textContent = 'Error: ' + fileName;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('file-dimensions').textContent = 'Error: ' + error;
    });
});
