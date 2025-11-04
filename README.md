# BergbauSchild ⚒️🛡️

BergbauSchild is an AI-powered platform designed to enhance mining safety in India by digitizing, analyzing, and interpreting extensive collections of mining accident records. Leveraging cutting-edge NLP, autonomous AI agents, and RAG, the system delivers actionable safety insights, real-time trends, and automated audit reports to improve hazard detection, root cause analysis, and regulatory compliance.

---

## Problem Statement

Mining accidents have historically been a major concern in India, necessitating robust safety measures and deep analytical capabilities for prevention. BergbauSchild addresses this challenge by utilizing Natural Language Processing (NLP) to digitize and analyze over 300 incident records from the Directorate General of Mines Safety (DGMS), India (2016–2022).

The platform offers:

- Efficient retrieval and analysis of accident data using advanced AI technologies.
- Interactive, real-time visualization of accident trends, locations, and timelines.
- Automated generation of detailed safety audit reports, reducing human effort while enhancing accuracy.
- Modular and adaptable architecture suitable for mines of varying sizes and complexities.

---

## Key Features

### AI-Powered Data Analysis & Visualization

- Ingests and processes historical mining accident data to extract essential patterns and insights.
- Displays real-time trends, geographical accident mapping, and chronological timelines through an intuitive user interface.

### Autonomous Safety Monitoring Agents

- Continuously scans new DGMS updates, mine inspection reports, and relevant local news.
- Automatically classifies incidents, flags hazards, and generates alerts such as _“Increase in transportation machinery accidents in Jharkhand mines in Q3 2022”_.
- Recommends targeted inspections and preventive measures to mine operators.

### Interactive Digital Mine Safety Officer

- Conversational agent layer built atop the NLP platform.
- Answers domain-specific queries such as _“Show me all methane-related accidents in 2021 in underground coal mines”_.
- Suggests regulatory compliance actions, e.g., _“Mine X exceeds threshold for ground movement incidents; schedule slope stability inspection”_.
- Returns accident-type codes based on user inputs, aligned with DGMS classification codes from the dataset.

### Adaptability & Extensibility

- Designed to easily incorporate new accident data beyond the initially provided 2015 training dataset.
- Capable of classifying new accidents and inferring accident types in real-time dialogue.

---

## Dataset

The dataset includes detailed accident reports from DGMS India, covering the years 2016 to 2022, with over 300 records capturing various mining accident types. The dataset provides accident reports, accident-type classification codes (from accompanying documentation), and additional metadata to support comprehensive analysis.

Dataset and code references used:

- Original PDF accident reports and extracted datasets.
- Accident classification codes described in the last pages of the DGMS reports.

---

## Getting Started

### Installation (Development)

Follow these steps to set up BergbauSchild for local development. The project consists of a backend (Python) and a frontend (Next.js).

#### 1. Clone the Repository

Clone this repository and navigate into the project directory:

```sh
git clone https://github.com/AkshatRai07/BergbauSchild.git
cd BergbauSchild
```


#### 2. Backend Setup

- Change into the backend directory:

```sh
cd backend
```

- Create and activate a Python virtual environment:

```sh
python -m venv venv
source venv/bin/activate    # On Windows: .\venv\Scripts\activate
```

- Install backend dependencies:

```sh
pip install -r requirements.txt
```

- Configure environment variables as needed (create a `.env` file):

```sh
cp .env.sample .env
# Edit .env with your dataset paths and API keys
```

- Start the backend server:

```sh
python app.py
```

#### 3. Frontend Setup

- In a new terminal, change to the frontend directory:

```sh
cd ../frontend
```

- Install frontend dependencies with Yarn:

```sh
yarn install
```

- Configure environment variables for the frontend (create a `.env` file):

```sh
cp .env.sample .env
```

- Launch the frontend development server:

```sh
yarn dev
```

#### 4. Access the Application

- The dashboard will be available at [http://localhost:3000](http://localhost:3000).
- The backend API will be available at [http://localhost:8000](http://localhost:8000) unless otherwise configured.

---

## Future Improvements

- Expand dataset coverage with latest DGMS records.
- Enhance AI models with incremental learning from new accident data.
- Integrate with other mining safety systems for holistic monitoring.
- Support multilingual queries and reports for wider accessibility.
