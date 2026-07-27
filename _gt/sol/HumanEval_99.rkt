#lang racket
(define (closest_integer value)
  (define x (string->number value))
  (cond [(>= x 0) (inexact->exact (floor (+ x 0.5)))]
        [else (inexact->exact (ceiling (- x 0.5)))]))
