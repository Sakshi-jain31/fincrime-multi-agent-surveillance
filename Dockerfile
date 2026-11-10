{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d0f6faef-e8f2-47c4-953b-9faec1af33f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Use an official secure, lightweight python micro-runtime\n",
    "FROM python:3.11-slim\n",
    "\n",
    "# Set internal systems environment directories\n",
    "WORKDIR /app\n",
    "\n",
    "# Install system curl tools required for basic endpoint container health monitoring\n",
    "RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*\n",
    "\n",
    "# Copy python dependency manifest definitions into workspace\n",
    "COPY requirements.txt .\n",
    "\n",
    "# Install compilation prerequisites without staging cache footprint bloat\n",
    "RUN pip install --no-cache-dir -r requirements.txt\n",
    "\n",
    "# Copy active code assets, static UI panels, and your updated Excel data sheet\n",
    "COPY app.py index.html historical_trades.xlsx ./\n",
    "\n",
    "# Create operational workspace directories for local logging targets\n",
    "RUN mkdir -p data\n",
    "\n",
    "# Expose internal execution socket port \n",
    "EXPOSE 8080\n",
    "\n",
    "# Execute server initialization by default\n",
    "CMD [\"python\", \"app.py\"]\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
