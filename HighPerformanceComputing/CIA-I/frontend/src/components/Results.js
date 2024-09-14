import React from 'react';
import '../styles/Results.css';
import { useLocation } from 'react-router-dom';

const Results = () => {
    const location = useLocation();
    const { image_base64, distances, dijkstra_time_ms } = location.state || {};

    return (
        <div className="result-page">
            <div className="image-container">
            <div className="image-title">Graph Visualization</div>
            <div className='image-display'>
                {image_base64 && <img src={`data:image/png;base64,${image_base64}`} alt="Graph Visualization" />}
            </div>
            </div>

            <div className="distances-container">
            <div className="distances-title">Shortest Distances</div>
            <div className='table-display'>
                {distances && (
                    <table border="1">
                        <thead>
                            <tr>
                                <th className='vertex-name'>Vertex</th>
                                <th className='distance-name'>Distance</th>
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
            </div>
            <div className="time-container">
            <div className="time-title">Elapsed Time</div>
            <div className='time-display'>
                <p>{dijkstra_time_ms ? `${dijkstra_time_ms} ms` : 'No data available'}</p>
                </div>
            </div>
        </div>
    );
};

export default Results;
