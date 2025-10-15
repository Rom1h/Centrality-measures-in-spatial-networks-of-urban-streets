# Official python image
FROM python:3.12

# Working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# to avoid it redownloading everytime we change main.py
COPY main.py .

# Default command
CMD ["python3", "main.py"]

