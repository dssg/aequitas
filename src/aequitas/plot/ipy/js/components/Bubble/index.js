import React from "react";
import PropTypes from "prop-types";

import BubbleCenter from "~/components/BubbleCenter";

import "./style.scss";

const propTypes = {
  centerShape: PropTypes.string.isRequired,
  color: PropTypes.string.isRequired,
  onClick: PropTypes.func.isRequired,
  size: PropTypes.number.isRequired,
  x: PropTypes.number.isRequired,
  y: PropTypes.number.isRequired,
};

function Bubble(props) {
  return (
    <g onClick={props.onClick} className="aequitas-bubble">
      <circle
        className="aequitas-bubble-area"
        cx={props.x}
        cy={props.y}
        r={props.size}
        fill={props.color}
      />
      <BubbleCenter
        x={props.x}
        y={props.y}
        fill={props.color}
        shape={props.centerShape}
      />
    </g>
  );
}

Bubble.propTypes = propTypes;
export default Bubble;
