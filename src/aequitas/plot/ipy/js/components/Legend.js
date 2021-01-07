import React from "react";
import sizes from "../enums/sizes";
import { highlightColor } from "../utils/colors";

import BubbleCenter from "./BubbleCenter";

export default function Legend(props) {
  const sortedGroups = [
    props.referenceGroup,
    ...props.groups.filter((item) => item !== props.referenceGroup),
  ];

  return (
    <div
      className="aequitas-legend"
      style={{ marginTop: sizes.MARGIN.top + 50 }}
    >
      <p className="aequitas-legend-title">Groups</p>
      <p>Click to highlight a group</p>
      <div className="aequitas-legend-items-group">
        {sortedGroups.map((group) => {
          let groupColor = props.scaleColor(group);

          if (group === props.hoverGroup) {
            groupColor = highlightColor(groupColor);
          }

          return (
            <div
              className="aequitas-legend-item"
              key={`legend-item-${group}`}
              onMouseEnter={() => props.handleHoverGroup(group)}
              onMouseLeave={() => props.handleHoverGroup(null)}
            >
              <svg width={10} height={10}>
                <BubbleCenter
                  x={5}
                  y={5}
                  shape={props.scaleShape(group)}
                  fill={groupColor}
                />
              </svg>
              <p style={{ color: groupColor }}>{group}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
