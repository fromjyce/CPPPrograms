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
    const navigate = useNavigate();

    return (
        <div className="main-screen">
        
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
