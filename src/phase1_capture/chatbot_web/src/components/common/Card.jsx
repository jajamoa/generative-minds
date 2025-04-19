import React from 'react';
import PropTypes from 'prop-types';
import './Card.css';

const Card = ({ children, className, ...props }) => {
  return (
    <div className={`card ${className || ''}`} {...props}>
      {children}
    </div>
  );
};

Card.propTypes = {
  children: PropTypes.node.isRequired,
  className: PropTypes.string,
};

export default Card; 