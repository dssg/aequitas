import React from "react";
import { Text } from "@vx/text";

import sizes from "../enums/sizes";

export default function MetricLine(props) {
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
