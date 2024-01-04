/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React from "react";
import PropTypes from "prop-types";
import { RiPushpin2Line as PinIcon, RiPushpin2Fill as UnpinIcon } from "react-icons/ri";

import { useAppDispatch, useAppState, useTunerState } from "~/utils/context";
import { ACTIONS } from "~/utils/reducer";
import { id } from "~/utils/models";

function PinButton({ model }) {
    const dispatch = useAppDispatch();
    const { pinnedModel } = useAppState();
    const { recommendedModel } = useTunerState();

    const isPinned = id(model) === id(pinnedModel);

    const handleClick = () => {
        if (!isPinned) {
            dispatch({ type: ACTIONS.PIN_MODEL, model });
        } else {
            dispatch({ type: ACTIONS.CLEAR_MODEL_PIN, recommendedModel });
        }
    };

    const buttonClassName = `button ${isPinned ? "unpin" : ""}`;
    const buttonText = `${isPinned ? "Unpin" : "Pin"} Model`;
    const Icon = isPinned ? UnpinIcon : PinIcon;

    return (
        <button className={buttonClassName} onClick={handleClick}>
            <Icon /> {buttonText}
        </button>
    );
}

PinButton.propTypes = {
    model: PropTypes.object.isRequired,
};

export default PinButton;
