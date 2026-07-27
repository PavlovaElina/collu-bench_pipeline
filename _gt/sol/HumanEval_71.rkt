#lang racket
(define (triangle_area a b c)
  (if (and (> (+ a b) c) (> (+ a c) b) (> (+ b c) a))
      (let* ([s (/ (+ a b c) 2.0)]
             [area (sqrt (* s (- s a) (- s b) (- s c)))])
        (/ (round (* area 100)) 100.0))
      -1))
