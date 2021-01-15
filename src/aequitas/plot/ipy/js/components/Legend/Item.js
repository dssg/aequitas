import React from "react";
import PropTypes from "prop-types";
import BubbleCenter from "~/components/BubbleCenter";

import { getGroupColorNew } from "~/utils/colors";

function Item(props) {
  const color = getGroupColorNew(
    props.group,
    props.activeGroup,
    props.scaleColor,
    props.hoverItem
  );
  return (
    <div
      className="aequitas-legend-item"
      onClick={() => props.handleSelect(props.group)}
      onMouseEnter={() => props.handleHover(props.group)}
      onMouseLeave={() => props.handleHover(null)}
    >
      <svg width={10} height={10}>
        <BubbleCenter
          x={5}
          y={5}
          shape={props.scaleShape(props.group)}
          fill={color}
        />
      </svg>
      <p style={{ color: color }}>
        {props.group}
        {props.index === 0 ? " [REF]" : null}
      </p>
    </div>
  );
}

Item.propTypes = {
  activeGroup: PropTypes.string,
  group: PropTypes.string,
  handleSelect: PropTypes.func,
  hoverItem: PropTypes.string,
  index: PropTypes.number,
  scaleColor: PropTypes.func,
  scaleShape: PropTypes.func,
  handleHover: PropTypes.func,
};

export default Item;
