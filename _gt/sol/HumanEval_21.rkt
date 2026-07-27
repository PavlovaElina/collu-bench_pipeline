#lang racket
(define (rescale_to_unit numbers)
  (define mn (apply min numbers))
  (define mx (apply max numbers))
  (map (lambda (x) (/ (- x mn) (- mx mn))) numbers))
