import React from "react";
import PropTypes from "prop-types";

const propTypes = {
  fairnessThreshold: PropTypes.number.isRequired,
  thresholdColor: PropTypes.string.isRequired,
  thresholdDisplayString: PropTypes.string.isRequired
};

export default function Tooltip(props) {
  return (
    <div
      className="aequitas-threshold-tooltip"
      style={{ color: props.thresholdColor }}
    >
      <p>
        Fairness Threshold
        <br />
        <span className="aequitas-bold-text">
          {props.thresholdDisplayString}
        </span>
      </p>{" "}
    </div>
  );
}

Tooltip.propTypes = propTypes;
