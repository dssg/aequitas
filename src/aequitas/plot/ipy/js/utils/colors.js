import { isNull } from "lodash";
import tinycolor from "tinycolor2";

export function highlight(color) {
  return tinycolor(color).brighten(15).toString();
}

function fade(color) {
  return tinycolor(color).desaturate(50).brighten(30).toString();
}

export function getGroupColor(group, activeGroup, scale) {
  if (group !== activeGroup && !isNull(activeGroup)) {
    return fade(scale(group));
  }

  return scale(group);
}
