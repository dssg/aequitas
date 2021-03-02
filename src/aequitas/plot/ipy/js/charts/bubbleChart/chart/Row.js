import React from "react";
import PropTypes from "prop-types";
import Tippy from "@tippyjs/react";
import onClickOutside from "react-onclickoutside";
import { filter } from "lodash";

import MetricLine from "~/components/MetricLine";
import Bubble from "~/components/Bubble";
import Tooltip from "./Tooltip";

import { getGroupColorNew } from "~/utils/colors";
import sizes from "~/constants/sizes";
import { fromPairs } from "lodash";

const propTypes = {
  data: PropTypes.array.isRequired,
  handleActiveGroup: PropTypes.func.isRequired,
  metric: PropTypes.string.isRequired,
  referenceGroup: PropTypes.string.isRequired,
  scaleBubbleSize: PropTypes.func.isRequired,
  scaleColor: PropTypes.func.isRequired,
  scalePosition: PropTypes.func.isRequired,
  scaleShape: PropTypes.func.isRequired,
  y: PropTypes.number.isRequired,
  activeGroup: PropTypes.string,
  dataColumnNames: PropTypes.object.isRequired
};

function Row(props) {
  Row.handleClickOutside = () => {
    return props.handleActiveGroup(null, true);
  };

  const relevantData = filter(
    props.data,
    (row) =>
      row[props.metric] >= props.axisBounds[0] &&
      row[props.metric] <= props.axisBounds[1]
  );

  return (
    <g height={sizes.ROW_HEIGHT}>
      <MetricLine
        y={props.y}
        metric={props.metric}
        lineStart={sizes.AXIS.LEFT.width}
        lineEnd={sizes.CHART_WIDTH - sizes.CHART_PADDING.right}
      />
      {relevantData.map((row) => {
        const groupName = row["attribute_value"];
        const groupColor = getGroupColorNew(
          groupName,
          props.activeGroup,
          props.scaleColor
        );

        return (
          <g key={`aequitas-bubble-$45{props.metric}-${groupName}`}>
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
                  x={props.scalePosition(
                    row[props.dataColumnNames[props.metric]]
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
  handleClickOutside: () => Row.handleClickOutside
};

Row.propTypes = propTypes;

export default onClickOutside(Row, clickOutsideConfig);
