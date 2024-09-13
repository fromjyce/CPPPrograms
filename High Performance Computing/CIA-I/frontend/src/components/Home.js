import React, { useEffect, useState } from 'react';

import '../styles/Home.css';

const SplashScreen = ({ onAnimationEnd, className }) => {
    return (
        <div className={`splash-screen ${className}`} onAnimationEnd={onAnimationEnd}>
            <h1>Dijkstra's Algorithm Implementation: Serial and Parallel Versions</h1>
        </div>
    );
};

const MainScreen = () => {
    const [numVertices, setNumVertices] = useState('');
    const [numEdges, setNumEdges] = useState('');
    const [edgesInput, setEdgesInput] = useState([]);
    const [initialVertex, setInitialVertex] = useState('');
    const [errorMessage, setErrorMessage] = useState('');

    const handleEdgesChange = (e) => {
        const edges = e.target.value;
        setNumEdges(edges);

        const edgesArray = Array.from({ length: edges }, (_, i) => ({
            startVertex: '',
            endVertex: '',
            edgeWeight: '0',
        }));

        setEdgesInput(edgesArray);
    };

    const handleEdgeInputChange = (index, field, value) => {
        const updatedEdges = edgesInput.map((edge, i) =>
            i === index ? { ...edge, [field]: value } : edge
        );
        setEdgesInput(updatedEdges);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setErrorMessage('');
        if (!numVertices || !numEdges || !initialVertex) {
            setErrorMessage('Please fill in all the input fields before submitting.');
            return;
        }

        if (Number(numVertices) <= 0 || Number(numEdges) <= 0) {
            setErrorMessage('Number of vertices and edges must be positive.');
            return;
        }

        if (initialVertex < 0) {
            setErrorMessage('Initial vertex cannot be negative.');
            return;
        }

        if (edgesInput.some(edge => !edge.startVertex || !edge.endVertex)) {
            setErrorMessage('Start and end vertices for each edge cannot be empty.');
            return;
        }

        if (edgesInput.some(edge => Number(edge.edgeWeight) < 0)) {
            setErrorMessage('Edge weights cannot be negative.');
            return;
        }

        const formDataObj = {
            vertices: numVertices,
            edges: numEdges,
            initial_vertex: initialVertex,
        };

        edgesInput.forEach((edge, i) => {
            formDataObj[`start_vertex_${i}`] = edge.startVertex;
            formDataObj[`end_vertex_${i}`] = edge.endVertex;
            formDataObj[`edge_weight_${i}`] = edge.edgeWeight;
        });

        try {
            const response = await fetch('http://127.0.0.1:5000/api/graph', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formDataObj),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
        } catch (error) {
            setErrorMessage('An error occurred while submitting the data.');
        }
    };

    return (
        <div className="dijkstra-container">
            <div className="left-side-container">
                <div className="dijkstra-info">
                    <h1 className='algorithm-info-title'><b>Dijkstra's Algorithm</b></h1>
                    <p className='algorithm-info-message'>
                        A Graph algorithm that finds the shortest path from a <strong>source vertex</strong> to all other vertices in the Graph (single source shortest path). It is a type of Greedy Algorithm that only works on Weighted Graphs having positive weights. The time complexity of Dijkstra's Algorithm is <strong>O(V<sup>2</sup>)</strong> with the help of the adjacency matrix representation of the graph.
                    </p>
                </div>
            </div>
            <div className="right-side-container">
                <h3 className='algorithm-title'>Enter the Number of Vertices and Edges</h3>
                <form onSubmit={handleSubmit} className="form-container">
                    <div>
                        <input
                            type="number"
                            name="vertices"
                            placeholder="Number of vertices"
                            value={numVertices}
                            onChange={(e) => setNumVertices(e.target.value)}
                        />
                    </div>
                    <div>
                        <input
                            type="number"
                            name="edges"
                            placeholder="Number of edges"
                            value={numEdges}
                            onChange={handleEdgesChange}
                        />
                    </div>
                    {numEdges > 0 && (
                        <>
                            <p className='enter-title'>Enter Start Vertex, End Vertex, and Edge Weight for each Edge</p>
                            {edgesInput.map((edge, index) => (
                                <div key={index} style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
                                    <input
                                        type="text"
                                        placeholder={`Start Vertex ${index + 1}`}
                                        value={edge.startVertex}
                                        onChange={(e) => handleEdgeInputChange(index, 'startVertex', e.target.value)}
                                    />
                                    <input
                                        type="text"
                                        placeholder={`End Vertex ${index + 1}`}
                                        value={edge.endVertex}
                                        onChange={(e) => handleEdgeInputChange(index, 'endVertex', e.target.value)}
                                    />
                                    <input
                                        type="number"
                                        placeholder={`Weight ${index + 1}`}
                                        value={edge.edgeWeight}
                                        onChange={(e) => handleEdgeInputChange(index, 'edgeWeight', e.target.value)}
                                    />
                                </div>
                            ))}
                        </>
                    )}
                    {numEdges > 0 && (
                        <>
                            <p className='enter-title'>Enter the Initial Vertex</p>
                            <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
                                <input
                                    type="text"
                                    name="initial_vertex"
                                    placeholder="Initial vertex"
                                    value={initialVertex}
                                    onChange={(e) => setInitialVertex(e.target.value)}
                                />
                            </div>
                        </>
                    )}
                    <button type="submit">Submit</button>
                    {errorMessage && <p className="error-message">{errorMessage}</p>}
                </form>
            </div>
        </div>
    );
};

const Home = () => {
    const [showSplash, setShowSplash] = useState(true);
    const [animationClass, setAnimationClass] = useState('');

    useEffect(() => {
        const timer = setTimeout(() => {
            setAnimationClass('exit');
        }, 2000);

        return () => clearTimeout(timer);
    }, []);

    const handleSplashAnimationEnd = () => {
        if (animationClass === 'exit') {
            setShowSplash(false);
        }
    };

    return (
        <div className="Home">
            {showSplash ? <SplashScreen onAnimationEnd={handleSplashAnimationEnd} className={animationClass} /> : <MainScreen />}
        </div>
    );
};

export default Home;
