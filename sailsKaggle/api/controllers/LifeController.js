/**
 * LifeController
 *
 * @description :: Server-side actions for handling incoming requests.
 * @help        :: See https://sailsjs.com/docs/concepts/actions
 */

var Meaning = require('the-ultimate-question');

module.exports = {
  purpose: function(req, res) {
    return res.json({
      answer: Meaning.answer(),
      question: Meaning.question()
    });
  }
};

