# Official python image
FROM python:3.12

# Working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY main.py .

# Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python3", "main.py"]

