import React, { useState } from "react";
import PropTypes from "prop-types";

import sizes from "~/constants/sizes";

import Item from "./Item";

import "./style.scss";

function Legend(props) {
  const [hoverItem, setHoverItem] = useState(null);

  return (
    <div
      className="aequitas-legend"
      style={{
        marginTop: sizes.AXIS.TOP.height + 50,
        marginLeft: sizes.LEGEND_MARGIN.left
      }}
    >
      <h1>Groups:</h1>
      <h2>Click to highlight a group</h2>
      <div>
        {props.groups.map((group, index) => (
          <Item
            key={`legend-item-${group}`}
            group={group}
            index={index}
            hoverItem={hoverItem}
            handleHover={setHoverItem}
            activeGroup={props.activeGroup}
            handleSelect={props.handleSelect}
            scaleShape={props.scaleShape}
            scaleColor={props.scaleColor}
          />
        ))}
      </div>
    </div>
  );
}

Legend.propTypes = {
  activeGroup: PropTypes.string,
  groups: PropTypes.array,
  handleSelect: PropTypes.func,
  scaleColor: PropTypes.func,
  scaleShape: PropTypes.func
};

export default Legend;
