window.onload = () => {
    loadPaperList();
    renderHistory();
};

/*
    Fetches available paper titles from FastAPI and populates the dropdown
*/
let globalPaperList = [];
async function loadPaperList() {
    try {
        const response = await fetch('http://127.0.0.1:8000/api/papers');
        const result = await response.json();

        if (response.ok && result.papers) {
            globalPaperList = result.papers; // Save to global variable
            generateDropdowns(); // Populate the initial dropdowns
        }
    } catch (error) {
        console.error("Failed to fetch paper titles:", error);
    }
}

/*
    Handles uploading single or multiple PDFs
*/
async function uploadPaper() {
    const fileInput = document.getElementById('fileInput');
    const uploadStatus = document.getElementById('uploadStatus');

    if (!fileInput.files.length) {
        alert("Please select at least one PDF file!");
        return;
    }

    const formData = new FormData();
    for (let i = 0; i < fileInput.files.length; i++) {
        formData.append('files', fileInput.files[i]);
    }

    uploadStatus.style.color = "#7f8c8d";
    uploadStatus.innerText = `Ingesting ${fileInput.files.length} paper(s)... Please wait.`;

    try {
        const response = await fetch('http://127.0.0.1:8000/api/ingest', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            uploadStatus.style.color = "#27ae60";
            uploadStatus.innerText = result.message;
            fileInput.value = ""; 
            
            // Refresh paper list dropdown with newly ingested papers
            loadPaperList(); 
        } else {
            uploadStatus.style.color = "#c0392b";
            uploadStatus.innerText = "Error: " + (result.detail || "Upload failed.");
        }
    } catch (error) {
        uploadStatus.style.color = "#c0392b";
        uploadStatus.innerText = "Failed to connect to backend server.";
        console.error("Upload error:", error);
    }
}

/*
    Sends question and selected paper filter to backend
*/
async function askCopilot() {
    const query = document.getElementById('queryInput').value;
    const selectedPaper = document.getElementById('paperSelect').value;
    const responseBox = document.getElementById('responseBox');
    const loadingIndicator = document.getElementById('loadingIndicator');

    if (!query) {
        alert("Please enter a question!");
        return;
    }

    loadingIndicator.style.display = "block";
    responseBox.innerText = "";

    try {
        const response = await fetch('http://127.0.0.1:8000/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                query: query,
                paper_title: selectedPaper 
            })
        });

        const result = await response.json();

        if (response.ok) {
            responseBox.innerText = result.data;
            // Save successful result to Local Storage
            saveToHistory(query, selectedPaper, result.data);
        } else {
            responseBox.innerText = "Error: " + (result.detail || "Query failed.");
        }
    } catch (error) {
        responseBox.innerText = "Failed to connect to the server. Is FastAPI running?";
        console.error("Fetch error:", error);
    } finally {
        loadingIndicator.style.display = "none";
    }
}

/*
    Saves Q&A interaction to browser local storage
*/
function saveToHistory(query, paper, response) {
    let history = JSON.parse(localStorage.getItem('researchHistory')) || [];
    
    const newInteraction = {
        query: query,
        paper: paper,
        response: response,
        timestamp: new Date().toLocaleString()
    };
    
    history.unshift(newInteraction); // Add newest entry to the top
    localStorage.setItem('researchHistory', JSON.stringify(history));
    renderHistory();
}

/*
    Renders history cards from browser local storage
*/
function renderHistory() {
    const historyContainer = document.getElementById('historyContainer');
    const history = JSON.parse(localStorage.getItem('researchHistory')) || [];
    
    historyContainer.innerHTML = "";
    
    if (history.length === 0) {
        historyContainer.innerHTML = "<p style='color: #7f8c8d; font-size: 14px;'>No history found. Ask a question to start saving!</p>";
        return;
    }
    
    history.forEach(item => {
        const card = document.createElement('div');
        card.className = 'history-card';
        
        card.innerHTML = `
            <div class="history-query">Q: ${item.query}</div>
            <div class="history-meta">Target: ${item.paper} | ${item.timestamp}</div>
            <div class="history-response">${item.response}</div>
        `;
        
        historyContainer.appendChild(card);
    });
}

/*
    Clears local storage history
*/
function clearHistory() {
    if(confirm("Are you sure you want to delete all saved research history?")) {
        localStorage.removeItem('researchHistory');
        renderHistory();
    }
}

function generateDropdowns() {
    const num = parseInt(document.getElementById('numPapers').value);
    const container = document.getElementById('dynamicDropdowns');
    
    // Clear current dropdowns
    container.innerHTML = '';

    for (let i = 0; i < num; i++) {
        const select = document.createElement('select');
        select.className = 'paper-select dropdown-select';
        
        // Always add the "ALL" option
        select.innerHTML = '<option value="ALL">All Ingested Papers</option>';
        
        // Add the specific papers
        globalPaperList.forEach(title => {
            const option = document.createElement('option');
            option.value = title;
            option.textContent = title;
            select.appendChild(option);
        });

        container.appendChild(select);
    }
}

// Master function to handle all AI buttons
async function runFeature(featureType) {
    const query = document.getElementById('queryInput').value;
    const responseBox = document.getElementById('responseBox');
    const loadingIndicator = document.getElementById('loadingIndicator');

    if (!query) {
        alert("Please enter a research topic or question first!");
        return;
    }

    // Gather all selected papers from the dynamic dropdowns
    const selectElements = document.querySelectorAll('.paper-select');
    const selectedPapers = Array.from(selectElements).map(select => select.value);

    loadingIndicator.style.display = "block";
    loadingIndicator.innerText = `Running ${featureType}... Please wait.`;
    responseBox.innerText = "";

    // Determine which API endpoint to call based on the button clicked
    // Inside runFeature(featureType) in frontend/script.js:

    let endpoint = "";
    let payload = { query: query };

    if (featureType === 'ask') {
        endpoint = "/api/ask";
        payload.paper_title = selectedPapers[0] || "ALL";
    } else if (featureType === 'timeline') {
        endpoint = "/api/analyze/timeline";
    } else if (featureType === 'compare') {
        endpoint = "/api/analyze/compare";
        
        // Filter unique paper selections, skipping 'ALL'
        const uniqueSelected = [...new Set(selectedPapers.filter(p => p !== "ALL"))];

        if (uniqueSelected.length < 2) {
            alert("Please select at least 2 distinct papers in the dropdowns for comparison.");
            loadingIndicator.style.display = "none";
            return;
        }

        payload = {
            topic: query,
            paper_titles: uniqueSelected
        };
    } else if (featureType === 'deep-dive') {
        endpoint = "/api/research/deep-dive";
    }

    try {
        const response = await fetch(`http://127.0.0.1:8000${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (response.ok) {
            responseBox.innerText = result.data;
            saveToHistory(`[${featureType.toUpperCase()}] ${query}`, selectedPapers.join(", "), result.data);
        } else {
            responseBox.innerText = "Error: " + (result.detail || "Query failed.");
        }
    } catch (error) {
        responseBox.innerText = "Failed to connect to the server.";
        console.error("Fetch error:", error);
    } finally {
        loadingIndicator.style.display = "none";
    }
}