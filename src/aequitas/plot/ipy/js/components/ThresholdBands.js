import React from "react";

export default function ThresholdBands(props) {
  function renderThresholdBand(side) {
    let x = props.scale.range()[0];
    let width = props.scale(-props.fairnessThreshold) - props.scale.range()[0];
    let ruleX = x + width;

    if (side === "right") {
      x = props.scale(props.fairnessThreshold);
      width = props.scale.range()[1] - props.scale(props.fairnessThreshold);
      ruleX = x;
    }
    return (
      <g key={`threshold-band-${side}`}>
        <rect
          className="aequitas-threshold-band"
          x={x}
          y={props.y}
          width={width}
          height={props.height}
          fill={props.color}
        />
        <line
          className="aequitas-threshold-rule"
          x1={ruleX}
          x2={ruleX}
          y1={props.y}
          y2={props.height}
          stroke={props.color}
        />
      </g>
    );
  }
  return <g>{[renderThresholdBand("left"), renderThresholdBand("right")]}</g>;
}
