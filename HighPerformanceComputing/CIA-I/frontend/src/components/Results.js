import React from 'react';
import '../styles/Results.css';

const Results = () => {
    return (
        <div className="result-page-container">
            {/* Division 1 */}
            <div className="result-division division-one">
                <h2>Graph Details</h2>
                <p>Display graph-related details or visualization here.</p>
            </div>
            
            {/* Division 2 */}
            <div className="result-division division-two">
                <h2>Algorithm Output</h2>
                <p>Show the results of Dijkstra's algorithm or any other output here.</p>
            </div>
            
            {/* Division 3 */}
            <div className="result-division division-three">
                <h2>Analysis and Summary</h2>
                <p>Provide additional analysis or summary details.</p>
            </div>
        </div>
    );
};

export default Results;
