import React from "react";
import PropTypes from "prop-types";
import { Text } from "@vx/text";

import "./style.scss";

const propTypes = {
  lineEnd: PropTypes.number.isRequired,
  lineStart: PropTypes.number.isRequired,
  metric: PropTypes.string.isRequired,
  y: PropTypes.number.isRequired,
};

function MetricLine(props) {
  return (
    <g>
      <Text
        x={0}
        y={props.y}
        className="aequitas-metric-label"
        verticalAnchor="middle"
      >
        {props.metric.toUpperCase()}
      </Text>
      <line
        x1={props.lineStart}
        y1={props.y}
        x2={props.lineEnd}
        y2={props.y}
        className="aequitas-metric-line"
      />
    </g>
  );
}

MetricLine.propTypes = propTypes;
export default MetricLine;
