import React from "react";

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
