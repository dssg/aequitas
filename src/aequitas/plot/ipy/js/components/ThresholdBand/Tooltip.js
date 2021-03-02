import React from "react";
import PropTypes from "prop-types";

const propTypes = {
  fairnessThreshold: PropTypes.number.isRequired,
  thresholdColor: PropTypes.string.isRequired,
  thresholdTooltipString: PropTypes.string.isRequired
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
          {props.thresholdTooltipString}
        </span>
      </p>{" "}
    </div>
  );
}

Tooltip.propTypes = propTypes;
