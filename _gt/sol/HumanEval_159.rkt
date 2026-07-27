#lang racket
(define (eat number need remaining)
  (if (>= remaining need)
      (list (+ number need) (- remaining need))
      (list (+ number remaining) 0)))
