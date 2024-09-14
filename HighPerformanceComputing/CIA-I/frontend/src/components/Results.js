import React from 'react';
import '../styles/Results.css';
import { useLocation } from 'react-router-dom';

const Results = () => {
    const location = useLocation();
    const { image_base64, distances, dijkstra_time_ms } = location.state || {};

    return (
        <div className="result-page">
            {/* First Division: Display Image */}
            <div className="image-container">
                <h2>Graph Visualization</h2>
                {image_base64 && <img src={`data:image/png;base64,${image_base64}`} alt="Graph Visualization" />}
            </div>

            {/* Second Division: Display Distance Table */}
            <div className="distances-container">
                <h2>Shortest Distances</h2>
                {distances && (
                    <table border="1">
                        <thead>
                            <tr>
                                <th>Vertex</th>
                                <th>Distance</th>
                            </tr>
                        </thead>
                        <tbody>
                            {Object.keys(distances).map((vertex) => (
                                <tr key={vertex}>
                                    <td>{vertex}</td>
                                    <td>{distances[vertex]}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>

            {/* Third Division: Display Elapsed Time */}
            <div className="time-container">
                <h2>Elapsed Time</h2>
                <p>{dijkstra_time_ms ? `${dijkstra_time_ms} ms` : 'No data available'}</p>
            </div>
        </div>
    );
};

export default Results;
