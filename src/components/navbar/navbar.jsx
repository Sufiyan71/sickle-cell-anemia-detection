import React from 'react';
import './navbar.css';

const Navbar = () => {
    return (
        <nav className="navbar">
            <ul className="navbar-menu">
                <li className="navbar-item"><a href="#home">HOME</a></li>
                <li className="navbar-item"><a href="#about">About</a></li>
                <li className="navbar-item"><a href="#contact">Contact</a></li>
                <li className="navbar-item"><a href="#records">Records</a></li>
            </ul>
        </nav>
    );
};

export default Navbar;
