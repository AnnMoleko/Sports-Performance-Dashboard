# 🏀 Sports Performance Dashboard

This project is a **Sports Performance Dashboard** built in **Power BI** to analyze NBA player and team performance using **wearable device data and game statistics**. The dashboard is divided into three main pages:

1. **Player Performance** – Radar charts and performance metrics for individual players.
2. **Team Strategy** – Aggregated team insights, shooting trends, and contributions.
3. **Predictive Insights** – Data-driven projections and advanced analytics.

---

## Features

* **Wearable Device Metrics**

  * Heart Rate
  * Speed
  * Distance
  * Acceleration

* **Game Statistics**

  * Shots Made & Attempted
  * Shooting Accuracy
  * Total Points

* **Visualizations**

  * **Radar Charts** for player performance profiles
  * **Bar & Line Graphs** for team strategies and trends
  * **Predictive Models** (using DAX & Power BI forecasting)

---

## Implementation

* **Data Collection**
  Synthetic NBA player dataset generated with columns:
  `Player, Team, Date, HeartRate, Speed, Distance, Acceleration, Shots Made, Shots Attempted, Points`.

* **Data Transformation**

  * Data cleaned and reshaped in **Power Query**.
  * Calculated Columns & Measures created using **DAX**:

    * Shooting Accuracy = `DIVIDE([Shots Made], [Shots Attempted], 0)`
    * Total Team Points = `SUM(Points)`
    * Player Contribution % = `DIVIDE([Player Points], [Team Total Points], 0)`

* **Dashboard Pages**

  * **Page 1: Player Performance** → Radar charts & player KPIs.
  * **Page 2: Team Strategy** → Aggregated metrics, team strengths, shooting trends.
  * **Page 3: Predictive Insights** → Forecasting & projections using Power BI.

---

## How to Use

1. Clone this repository.
2. Open the `.pbix` file in **Power BI Desktop**.
3. Explore the dashboard pages:

   * Player Performance
   * Team Strategy
   * Predictive Insights

---

## Repository Structure

```
📁 Sports-Performance-Dashboard
 ┣ 📊 Sports_Performance_Dataset.xlsx   # Sample dataset  
 ┣ 📄 Sports_Performance_Dashboard.pbix # Power BI report  
 ┣ 📁 images/                           # Dashboard screenshots  
 ┃ ┣ player_performance.png  
 ┃ ┣ team_strategy.png  
 ┃ ┗ predictive_insights.png  
 ┣ 📄 README.md                         # Project documentation  
 
```

---

## Future Improvements

* Integration with **real-time wearable device APIs**.
* More advanced **predictive analytics using Python/R scripts** inside Power BI.
* Player comparison reports and injury risk tracking.

---

## 👨Author

Developed by \[Nthabeleng Moleko]
📧 Contact: \[nthabelengmoleko0211@gmail.com]
🔗 GitHub: \[https://github.com/AnnMoleko]


