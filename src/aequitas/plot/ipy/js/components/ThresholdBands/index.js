import React from "react";
import PropTypes from "prop-types";
import Tippy from "@tippyjs/react";

import Tooltip from "./Tooltip";
import "./style.scss";

const propTypes = {
  color: PropTypes.string.isRequired,
  fairnessThreshold: PropTypes.number.isRequired,
  height: PropTypes.number.isRequired,
  scale: PropTypes.func.isRequired,
  y: PropTypes.number.isRequired,
};

function ThresholdBands(props) {
  function renderThresholdBand(side) {
    let x, width, ruleX;

    if (side === "left") {
      x = props.scale.range()[0];
      width =
        props.scale(-props.fairnessThreshold + 1) - props.scale.range()[0];
      ruleX = x + width;
    } else if (side === "right") {
      x = props.scale(props.fairnessThreshold - 1);
      width = props.scale.range()[1] - props.scale(props.fairnessThreshold - 1);
      ruleX = x;
    } else {
      return null;
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
        <Tippy
          content={
            <Tooltip
              fairnessThreshold={props.fairnessThreshold}
              color={props.color}
            />
          }
          placement="top"
        >
          <line
            className="aequitas-threshold-rule"
            x1={ruleX}
            x2={ruleX}
            y1={props.y}
            y2={props.height}
            stroke={props.color}
          />
        </Tippy>
      </g>
    );
  }
  return <g>{[renderThresholdBand("left"), renderThresholdBand("right")]}</g>;
}

ThresholdBands.propTypes = propTypes;
export default ThresholdBands;
