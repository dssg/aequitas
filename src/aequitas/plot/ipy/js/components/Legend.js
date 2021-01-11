import React, { useState } from "react";
import PropTypes from "prop-types";
import sizes from "../enums/sizes";
import { getGroupColor, highlight } from "../utils/colors";

import BubbleCenter from "./BubbleCenter";

const propTypes = {
  activeGroup: PropTypes.string,
  groups: PropTypes.array.isRequired,
  handleActiveGroup: PropTypes.func.isRequired,
  referenceGroup: PropTypes.string.isRequired,
  scaleColor: PropTypes.func.isRequired,
  scaleShape: PropTypes.func.isRequired,
};

function Legend(props) {
  const [hoverItem, setHoverItem] = useState(null);
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
          let groupColor;

          if (group === hoverItem) {
            groupColor = highlight(props.scaleColor(group));
          } else {
            groupColor = getGroupColor(
              group,
              props.activeGroup,
              props.scaleColor
            );
          }

          return (
            <div
              className="aequitas-legend-item"
              key={`legend-item-${group}`}
              onClick={() => props.handleActiveGroup(group)}
              onMouseEnter={() => setHoverItem(group)}
              onMouseLeave={() => setHoverItem(null)}
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

Legend.propTypes = propTypes;
export default Legend;
