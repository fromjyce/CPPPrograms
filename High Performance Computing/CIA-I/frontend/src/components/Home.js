import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
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
    const [terminalVertex, setTerminalVertex] = useState('');
    const [imageSrc, setImageSrc] = useState('');
    const [errorMessage, setErrorMessage] = useState('');
    const [algorithmName] = useState('british_museum_search');
    const leftLink = "/blind-algorithm-list";
    const rightLink = "/depth-first-search";

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

        if (!numVertices || !numEdges || edgesInput.some(edge => !edge.startVertex || !edge.endVertex) || !initialVertex || !terminalVertex) {
            setErrorMessage('Please fill in all the input fields before submitting.');
            return;
        }

        const formDataObj = {
            algorithm_name: algorithmName,
            vertices: numVertices,
            edges: numEdges,
            initial_vertex: initialVertex,
            terminal_vertex: terminalVertex,
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
        if (data.image_base64) {
            setImageSrc(`data:image/png;base64,${data.image_base64}`);
            setErrorMessage('');
        } else {
            setImageSrc('');
            setErrorMessage('Failed to retrieve image');
        }
        } catch (error) {
            setImageSrc('');
            setErrorMessage('An error occurred while fetching the image.');
        }
    };

    return (
        <div className="british-museum-search-container">
            <div className="left-side-container">
                <h1 className='algorithm-title'>British Museum Search</h1>
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
                                disabled
                            />
                        </div>
                    ))}
                    {numEdges > 0 && (
                        <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
                            <input
                                type="text"
                                name="initial_vertex"
                                placeholder="Initial vertex"
                                value={initialVertex}
                                onChange={(e) => setInitialVertex(e.target.value)}
                            />
                            <input
                                type="text"
                                name="terminal_vertex"
                                placeholder="Terminal vertex"
                                value={terminalVertex}
                                onChange={(e) => setTerminalVertex(e.target.value)}
                            />
                        </div>
                    )}
                    <button type="submit">Submit</button>
                    {errorMessage && <p className="error-message">{errorMessage}</p>}
                </form>
            </div>
            <div className="right-side-container">
            <div className="image-container">
                {imageSrc && <img src={imageSrc} alt="Graph visualization" />}
                {errorMessage && <p className="error-message">{errorMessage}</p>}
                </div>
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
