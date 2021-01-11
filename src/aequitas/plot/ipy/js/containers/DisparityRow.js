import React from "react";
import PropTypes from "prop-types";
import Tippy from "@tippyjs/react";
import onClickOutside from "react-onclickoutside";

import MetricLine from "../components/MetricLine";
import Bubble from "../containers/Bubble";
import { getGroupColor } from "../utils/colors";
import sizes from "../enums/sizes";
import Tooltip from "../components/Tooltip";

const propTypes = {
  data: PropTypes.array.isRequired,
  handleActiveGroup: PropTypes.func.isRequired,
  metric: PropTypes.string.isRequired,
  referenceGroup: PropTypes.string.isRequired,
  scaleBubbleSize: PropTypes.func.isRequired,
  scaleColor: PropTypes.func.isRequired,
  scaleDisparity: PropTypes.func.isRequired,
  scaleShape: PropTypes.func.isRequired,
  y: PropTypes.number.isRequired,
  activeGroup: PropTypes.string,
};

function DisparityRow(props) {
  DisparityRow.handleClickOutside = () => {
    return props.handleActiveGroup(null, true);
  };
  return (
    <g height={sizes.ROW_HEIGHT}>
      <MetricLine
        y={props.y}
        metric={props.metric}
        lineStart={sizes.MARGIN.left}
        lineEnd={sizes.WIDTH - sizes.MARGIN.right}
      />
      {props.data.map((row) => {
        const groupName = row["attribute_value"];
        const groupColor = getGroupColor(
          groupName,
          props.activeGroup,
          props.scaleColor
        );

        return (
          <g key={`aequitas-bubble-${props.metric}-${groupName}`}>
            <Tippy
              content={
                <Tooltip
                  data={row}
                  metric={props.metric}
                  isReferenceGroup={props.referenceGroup === groupName}
                  groupColor={props.scaleColor(groupName)}
                />
              }
              placement="right"
            >
              <g>
                <Bubble
                  x={props.scaleDisparity(
                    row[`${props.metric}_disparity_scaled`]
                  )}
                  y={props.y}
                  size={props.scaleBubbleSize(row["group_size"])}
                  color={groupColor}
                  centerShape={props.scaleShape(groupName)}
                  onClick={() => props.handleActiveGroup(groupName)}
                />
              </g>
            </Tippy>
          </g>
        );
      })}
    </g>
  );
}

const clickOutsideConfig = {
  handleClickOutside: () => DisparityRow.handleClickOutside,
};

DisparityRow.propTypes = propTypes;

export default onClickOutside(DisparityRow, clickOutsideConfig);
