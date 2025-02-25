let latestTaskPayload = null; // Store the last generated task payload

async function sendMessage() {
    let userInput = document.getElementById("user-input").value.trim().toLowerCase(); // Normalize input
    if (!userInput) return;
    console.log("User Input:", userInput);

    let chatBox = document.getElementById("chat-box");

    // Append user message (aligned to the right)
    let userMessage = document.createElement("div");
    userMessage.classList.add("user-message");
    userMessage.textContent = userInput;
    chatBox.appendChild(userMessage);

    document.getElementById("user-input").value = ""; // Clear input field
    chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the latest message

    let botMessage = document.createElement("div");
    botMessage.classList.add("bot-message");

    if (["hi", "hello", "hey", "good morning", "good evening", "good afternoon"].includes(userInput)) {
        botMessage.textContent = "Assistant: Hello! How can I assist you today? 😊";
        chatBox.appendChild(botMessage);
        chatBox.scrollTop = chatBox.scrollHeight;
        return;
    }

    // Case 1: User says "no" (cancel task)
    if (userInput === "no" && latestTaskPayload) {
        botMessage.textContent = "Assistant: Okay, task creation has been canceled.";
        chatBox.appendChild(botMessage);
        chatBox.scrollTop = chatBox.scrollHeight;
        latestTaskPayload = null; // Clear stored task
        return;
    }

    // Case 2: User confirms task creation with "yes"
    if (userInput === "yes" && latestTaskPayload) {
        console.log("User confirmed task creation. Sending API request...");

        let createTaskResponse = await fetch("/create-task", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(latestTaskPayload) // Send stored task payload
        });

        let taskResponseData = await createTaskResponse.json();
        console.log("Task Creation Response:", taskResponseData);

        if (taskResponseData && taskResponseData.task_details && taskResponseData.task_details.StatusCode === 200) {
            botMessage.innerHTML = `Assistant:
                <strong>Task Successfully Created!</strong><br>
                <strong>Task ID:</strong> ${taskResponseData.task_details.id || "N/A"} <br>
                <strong>Message:</strong> ${taskResponseData.task_details.message || "Task created successfully!"}
            `;
        } else {
            botMessage.innerHTML = `Assistant: <strong>Failed to create task.</strong><br>${taskResponseData.task_details?.ErrorMsg || "Unknown error"}`;
        }

        chatBox.appendChild(botMessage);
        chatBox.scrollTop = chatBox.scrollHeight;
        latestTaskPayload = null; // Reset stored payload after task creation
        return;
    }

    // Case 3: Process normal chat messages
    let response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_input: userInput })
    });

    let data = await response.json();
    console.log("data:", data);

    if (data.status === "waiting_for_answer" && data.question) {
        let cleanQuestion = data.question.replace(/^\d+\.\s*/, '');
        botMessage.textContent = cleanQuestion;

        if (cleanQuestion.includes("What is the category of this task?")) {
            console.log("Fetching categories...");
            let categoryResponse = await fetch("/fetch_categories");
            let categoryData = await categoryResponse.json();
            console.log("Category Data:", categoryData);

            if (categoryData.categories && categoryData.categories.length > 0) {
                botMessage.innerHTML += "<br><strong>Available Categories:</strong><br>";

                let ul = document.createElement("ul");
                categoryData.categories.forEach(category => {
                    let li = document.createElement("li");
                    li.textContent = category;
                    ul.appendChild(li);
                });

                botMessage.appendChild(ul);
            } else {
                botMessage.innerHTML += "<br>No categories found.";
            }
        }

        else if (cleanQuestion.includes("What is the sub-category of this task?")) {
            console.log("Fetching subcategories for category:", userInput);
            let subCategoryResponse = await fetch(`/fetch_subcategories/${userInput}`);
            let subCategoryData = await subCategoryResponse.json();
            console.log("Subcategory Data:", subCategoryData);

            if (subCategoryData.subcategories && subCategoryData.subcategories.length > 0) {
                botMessage.innerHTML += "<br><strong>Available Subcategories:</strong><br>";

                let ul = document.createElement("ul");
                subCategoryData.subcategories.forEach(subcategory => {
                    let li = document.createElement("li");
                    li.textContent = `ID: ${subcategory.id}, Name: ${subcategory.name}`;
                    ul.appendChild(li);
                });

                botMessage.appendChild(ul);
            } else {
                botMessage.innerHTML += "<br>No subcategories found.";
            }
        }
    }

    else if (data.status === "success") {
        if (data.task_payload) {
            latestTaskPayload = data.task_payload; // Store the task payload for confirmation
            
            botMessage.innerHTML = `Assistant:
                <strong>Task Created Successfully!</strong><br>
                <strong>Title:</strong> ${data.task_payload.title} <br>
                <strong>Requested By:</strong> ${data.task_payload.requested_by} <br>
                <strong>Message:</strong> ${data.task_payload.message} <br>
                <strong>Category ID:</strong> ${data.task_payload.category_id} <br>
                <strong>Sub Category ID:</strong> ${data.task_payload.sub_category_id} <br>
                <br><p>Would you like to create a ticket with these details? (Reply "yes" to confirm)</p>
            `;
        } else if (data.response) {
            let cleanResponse = data.response.replace(/\n/g, "<br>");
            botMessage.innerHTML = `<strong>Assistant:</strong> ${cleanResponse}`;
        } else {
            botMessage.textContent = "Unexpected response format";
        }
    }

    else {
        botMessage.textContent = "Error processing response";
    }

    chatBox.appendChild(botMessage);
    chatBox.scrollTop = chatBox.scrollHeight;
}

document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        event.preventDefault(); // Prevent the default form submission (if any)
        sendMessage();
    }
});