import React from "react";
import PropTypes from "prop-types";
import { Text } from "@vx/text";

const propTypes = {
  lineEnd: PropTypes.number.isRequired,
  lineStart: PropTypes.number.isRequired,
  metric: PropTypes.string.isRequired,
  y: PropTypes.number.isRequired,
};

function MetricLine(props) {
  return (
    <g>
      <Text x={0} y={props.y} className="aequitas-row-names">
        {props.metric.toUpperCase()}
      </Text>
      <line
        x1={props.lineStart}
        y1={props.y}
        x2={props.lineEnd}
        y2={props.y}
        className="aequitas-row-line"
      />
    </g>
  );
}

MetricLine.propTypes = propTypes;
export default MetricLine;
