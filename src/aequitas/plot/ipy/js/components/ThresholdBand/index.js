import React from "react";
import PropTypes from "prop-types";
import Tippy from "@tippyjs/react";

import Tooltip from "./Tooltip";
import "./style.scss";

const propTypes = {
  thresholdColor: PropTypes.string.isRequired,
  fairnessThreshold: PropTypes.number.isRequired,
  svgKeySuffix: PropTypes.string.isRequired,
  x: PropTypes.number.isRequired,
  ruleX: PropTypes.number.isRequired,
  y1: PropTypes.number.isRequired,
  y2: PropTypes.number.isRequired,
  width: PropTypes.number.isRequired,
  height: PropTypes.number.isRequired,
  thresholdDisplayString: PropTypes.string.isRequired
};

function ThresholdBand(props) {
  return (
    <g
      key={`threshold-band-${props.svgKeySuffix}`}
      className={`threshold-band-${props.svgKeySuffix}`}
    >
      <rect
        className="aequitas-threshold-band"
        x={props.x}
        y={props.y1}
        width={props.width}
        height={props.height}
        fill={props.thresholdColor}
      />
      <Tippy
        content={
          <Tooltip
            thresholdDisplayString={props.thresholdDisplayString}
            fairnessThreshold={props.fairnessThreshold}
            thresholdColor={props.thresholdColor}
          />
        }
        placement="top"
      >
        <line
          className="aequitas-threshold-rule"
          x1={props.ruleX}
          x2={props.ruleX}
          y1={props.y1}
          y2={props.y2}
          stroke={props.thresholdColor}
        />
      </Tippy>
    </g>
  );
}

ThresholdBand.propTypes = propTypes;
export default ThresholdBand;
