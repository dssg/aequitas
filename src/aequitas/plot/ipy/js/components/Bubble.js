import React from "react";
import BubbleCenter from "./BubbleCenter";

export default function Bubble(props) {
  return (
    <g
      onMouseEnter={() => props.handleHoverGroup(props.group)}
      onMouseLeave={() => props.handleHoverGroup(null)}
    >
      <circle
        className="aequitas-bubble"
        cx={props.x}
        cy={props.y}
        r={props.size}
        fill={props.color}
      />
      <BubbleCenter
        x={props.x}
        y={props.y}
        fill={props.color}
        shape={props.centerShape}
      />
    </g>
  );
}
