import React from "react";
import PropTypes from "prop-types";

const propTypes = {
    fairnessThreshold: PropTypes.number.isRequired,
    color: PropTypes.string.isRequired,
  };

export default function Tooltip(props) {
  return (
    <div className="aequitas-threshold-tooltip" style={{ color: props.color }}>
      <p>
        Fairness Threshold
        <br />
        <span className="aequitas-bold-text"> {props.fairnessThreshold}</span>
      </p>{" "}
    </div>
  );
}

Tooltip.propTypes = propTypes;