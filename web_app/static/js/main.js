document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('video-upload');
    const fileInfo = document.getElementById('file-info');
    const filename = document.getElementById('filename');
    const removeFileBtn = document.getElementById('remove-file');
    const submitBtn = document.getElementById('submit-btn');
    const loading = document.getElementById('loading');

    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        uploadArea.classList.add('highlight');
    }

    function unhighlight() {
        uploadArea.classList.remove('highlight');
    }

    // Handle file drop
    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            updateFileInfo();
        }
    }

    // Handle file selection via browse button
    fileInput.addEventListener('change', updateFileInfo);

    function updateFileInfo() {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            
            // Check file type
            const fileType = file.type;
            const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
            const validExtensions = ['.mp4', '.avi', '.mov', '.mkv'];
            
            let valid = false;
            
            // Check by MIME type
            if (validTypes.includes(fileType)) {
                valid = true;
            } else {
                // Check by extension as fallback
                const extension = '.' + file.name.split('.').pop().toLowerCase();
                if (validExtensions.includes(extension)) {
                    valid = true;
                }
            }
            
            if (!valid) {
                alert('Invalid file type. Please upload MP4, AVI, MOV, or MKV files only.');
                resetFileInput();
                return;
            }
            
            // Check file size (max 100MB)
            if (file.size > 100 * 1024 * 1024) {
                alert('File too large. Maximum size is 100MB.');
                resetFileInput();
                return;
            }
            
            // Update UI to show selected file
            filename.textContent = file.name;
            fileInfo.classList.remove('d-none');
            submitBtn.disabled = false;
        } else {
            resetFileInput();
        }
    }

    // Remove selected file
    removeFileBtn.addEventListener('click', resetFileInput);

    function resetFileInput() {
        fileInput.value = '';
        fileInfo.classList.add('d-none');
        filename.textContent = 'No file selected';
        submitBtn.disabled = true;
    }

    // Show loading spinner on form submit
    uploadForm.addEventListener('submit', function() {
        loading.classList.remove('d-none');
        submitBtn.disabled = true;
    });
});