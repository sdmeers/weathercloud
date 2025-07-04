# WeatherCloud

This repository contains a collection of services for a personal weather station project, designed to be deployed on Google Cloud Platform (GCP). It ingests data from a Pimoroni Enviro Weather device, stores it, and provides multiple interfaces for viewing and analyzing the data.

## Features

*   **Data Ingestion**: Receives weather data via an HTTP endpoint.
*   **Data Storage**: Stores time-series weather data in a Firestore NoSQL database.
*   **Web Dashboard**: A responsive web page to view the latest weather conditions.
*   **Interactive Analysis**: An interactive dashboard for analyzing historical data over custom time periods.
*   **Natural Language Queries**: A chatbot interface (powered by Gemini) to ask questions about the weather data.
*   **Image Classification**: A service to classify weather images into categories like sunny, cloudy, raining, etc.

## Repository Structure

The project is organized into several microservices:

| Directory                  | Description                                                                                                                            |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `get-weather-data`         | A service to retrieve weather data from an external source.                                                                            |
| `store-weather-data`       | A GCP Cloud Run function that provides an HTTP endpoint to accept and store readings from the Enviro Weather device in Firestore.        |
| `display-weather-data`     | A service to display the collected weather data.                                                                                       |
| `interactive_dashboard`    | A Dash-based interactive dashboard for detailed data analysis.                                                                         |
| `weather-chat`             | A chatbot application that uses a Large Language Model (Gemini) to answer natural language questions about the weather data.             |
| `weather-image-classifier` | A service that uses a Gemini 2.0 Flash Lite model to classify weather images.                                                          |
| `weather-dashboard`        | A Flask-based web application that serves a responsive dashboard to view current and historical weather data.                            |
| `kill-switch`              | A utility to stop or disable services.                                                                                                 |

## Deployment

The services are designed to be deployed as individual Cloud Run functions on GCP. Deployment can be done using the `gcloud` CLI. While detailed, user-specific instructions are not provided, each service's directory contains a `requirements.txt` and some have a `Dockerfile` to facilitate containerization and deployment.

## Screenshots

### Main Dashboard
![Screenshot of the web interface displaying the weather data including current temperature, humidity, pressure and more.](https://github.com/sdmeers/weathercloud/blob/main/screenshots/weatherstation-full.jpg)

### Interactive Analysis
![Screenshot of the interactive dashboard enabling detailed analysis temperature, humidity, pressure and more for a given date range.](https://github.com/sdmeers/weathercloud/blob/main/screenshots/dashboard_screenshot.jpg)

### Weather Chatbot
![Screenshot of the weather chatbot.](https://github.com/sdmeers/weathercloud/blob/main/screenshots/chatbot.jpg)

### Image Classifier
![Screenshot of the Image Classifier.](https://github.com/sdmeers/weathercloud/blob/main/screenshots/image_classifier.jpg)
