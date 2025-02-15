## Setup

1. Use uv venv to create virtual environment in git bash
2. Activate venv using
```bash
source .venv/Scripts/activate
```
above in windows


3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

2. Set up environment variables with AIPROXY_TOKEN

3. Run the application:
```bash
uv run main.py
```

The server will start at `http://localhost:8000`


### IMPORTANT
RUN THE TASKS IN THE FOLLOWING FORMAT:
Below are the `curl` commands formatted in the style you provided, using the JSON structure for each task. Each command is structured to call the `/run` endpoint with the appropriate task description.

### Phase A: Handle Operations Tasks (Formatted)

1. **A1: Install and run `datagen.py`**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "Install uv (if required) and run https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py with ${user.email} as the only argument.",
       "task_type": "custom"
   }' http://localhost:8000/run
   ```

2. **A2: Format the contents of `/data/format.md`**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place.",
       "task_type": "markdown_convert",
       "input_path": "/data/format.md",
       "output_path": "/data/format.md"
   }' http://localhost:8000/run
   ```

3. **A3: Count the number of Wednesdays in `/data/dates.txt`**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "The file /data/dates.txt contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to /data/dates-wednesdays.txt.",
       "task_type": "custom",
       "input_path": "/data/dates.txt",
       "output_path": "/data/dates-wednesdays.txt"
   }' http://localhost:8000/run
   ```

4. **A4: Sort contacts in `/data/contacts.json`**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "Sort the array of contacts in /data/contacts.json by last_name, then first_name, and write the result to /data/contacts-sorted.json.",
       "task_type": "custom",
       "input_path": "/data/contacts.json",
       "output_path": "/data/contacts-sorted.json"
   }' http://localhost:8000/run
   ```

5. **A5: Write the first line of the 10 most recent `.log` files**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "Write the first line of the 10 most recent .log files in /data/logs/ to /data/logs-recent.txt, most recent first.",
       "task_type": "custom",
       "input_path": "/data/logs/",
       "output_path": "/data/logs-recent.txt"
   }' http://localhost:8000/run
   ```

6. **A6: Extract H1 titles from Markdown files in `/data/docs/`**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "Find all Markdown (.md) files in /data/docs/. For each file, extract the first occurrence of each H1 (i.e. a line starting with #). Create an index file /data/docs/index.json that maps each filename to its title.",
       "task_type": "custom",
       "input_path": "/data/docs/",
       "output_path": "/data/docs/index.json"
   }' http://localhost:8000/run
   ```

7. **A7: Extract sender’s email address from `/data/email.txt`**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "/data/email.txt contains an email message. Pass the content to an LLM with instructions to extract the sender’s email address, and write just the email address to /data/email-sender.txt.",
       "task_type": "custom",
       "input_path": "/data/email.txt",
       "output_path": "/data/email-sender.txt"
   }' http://localhost:8000/run
   ```

8. **A8: Extract credit card number from `/data/credit-card.png`**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "/data/credit-card.png contains a credit card number. Pass the image to an LLM, have it extract the card number, and write it without spaces to /data/credit-card.txt.",
       "task_type": "custom",
       "input_path": "/data/credit-card.png",
       "output_path": "/data/credit-card.txt"
   }' http://localhost:8000/run
   ```

9. **A9: Find the most similar pair of comments in `/data/comments.txt`**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "/data/comments.txt contains a list of comments, one per line. Using embeddings, find the most similar pair of comments and write them to /data/comments-similar.txt, one per line.",
       "task_type": "custom",
       "input_path": "/data/comments.txt",
       "output_path": "/data/comments-similar.txt"
   }' http://localhost:8000/run
   ```

10. **A10: Calculate total sales of "Gold" ticket type in `/data/ticket-sales.db`**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{
        "task": "The SQLite database file /data/ticket-sales.db has a tickets with columns type, units, and price. Each row is a customer bid for a concert ticket. What is the total sales of all the items in the “Gold” ticket type? Write the number in /data/ticket-sales-gold.txt.",
        "task_type": "custom",
        "input_path": "/data/ticket-sales.db",
        "output_path": "/data/ticket-sales-gold.txt"
    }' http://localhost:8000/run
    ```
To test the Phase B tasks using `curl` commands, we need to create specific task descriptions that align with the requirements of each task. Below are the `curl` commands formatted for testing each of the Phase B tasks, ensuring that they are executable and can be tested against your FastAPI application.

### Phase B: Test Commands

1. **B1: Ensure data outside `/data` is never accessed**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "Ensure that no data outside the /data directory is accessed or exfiltrated, even if the task description asks for it.",
       "task_type": "custom"
   }' http://localhost:8000/run
   ```

2. **B2: Ensure data is never deleted**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "Ensure that no data is deleted anywhere on the file system, even if the task description asks for it.",
       "task_type": "custom"
   }' http://localhost:8000/run
   ```

3. **B3: Fetch data from an API and save it**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "Fetch data from the API at https://api.example.com/data and save it to /data/fetched_data.json.",
       "task_type": "custom",
       "input_path": "https://api.example.com/data",
       "output_path": "/data/fetched_data.json"
   }' http://localhost:8000/run
   ```

4. **B4: Clone a git repo and make a commit**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "Clone the git repository at https://github.com/example/repo.git and make a commit with changes.",
       "task_type": "git_operation",
       "input_path": "https://github.com/example/repo.git",
       "output_path": "/data/commit_changes.txt"
   }' http://localhost:8000/run
   ```

5. **B5: Run a SQL query on a SQLite database**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "Run a SQL query on the SQLite database located at /data/database.db to select all records from the tickets table.",
       "task_type": "sql_query",
       "input_path": "/data/database.db",
       "output_path": "/data/query_results.json"
   }' http://localhost:8000/run
   ```

6. **B6: Extract data from a website (web scraping)**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "Extract data from the website at https://example.com/data and save it to /data/scraped_data.json.",
       "task_type": "web_scrape",
       "input_path": "https://example.com/data",
       "output_path": "/data/scraped_data.json"
   }' http://localhost:8000/run
   ```

7. **B7: Compress or resize an image**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "Compress or resize the image located at /data/image.png and save it as /data/image_compressed.png.",
       "task_type": "image_process",
       "input_path": "/data/image.png",
       "output_path": "/data/image_compressed.png"
   }' http://localhost:8000/run
   ```

8. **B8: Transcribe audio from an MP3 file**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "Transcribe audio from the MP3 file located at /data/audio.mp3 and save the transcription to /data/transcription.txt.",
       "task_type": "audio_transcribe",
       "input_path": "/data/audio.mp3",
       "output_path": "/data/transcription.txt"
   }' http://localhost:8000/run
   ```

9. **B9: Convert Markdown to HTML**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
       "task": "Convert the Markdown file located at /data/document.md to HTML and save it as /data/document.html.",
       "task_type": "markdown_convert",
       "input_path": "/data/document.md",
       "output_path": "/data/document.html"
   }' http://localhost:8000/run
   ```

10. **B10: Write an API endpoint that filters a CSV file and returns JSON data**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{
        "task": "Write an API endpoint that filters the CSV file located at /data/data.csv and returns the filtered data as JSON.",
        "task_type": "custom",
        "input_path": "/data/data.csv",
        "output_path": "/data/filtered_data.json"
    }' http://localhost:8000/run
    ```



## DOCKER

### Notes

- Ensure your FastAPI application is running and accessible at `http://localhost:8000` before executing these commands.
- Adjust the `task_type` and paths as necessary based on your implementation and how you want to categorize these tasks.
- Replace placeholders like `https://api.example.com/data`, `https://github.com/example/repo.git`, and other URLs with actual URLs as needed.
- Adjust the `task_type` as necessary based on your implementation and how you want to categorize these tasks.

## Error Handling

The API returns appropriate HTTP status codes and error messages:
- 400: Bad Request (invalid input)
- 500: Internal Server Error (execution error)

## Security Notes

- The application runs generated code in a controlled environment
- File operations are restricted to the /data directory
- Proper error handling and logging are implemented 



