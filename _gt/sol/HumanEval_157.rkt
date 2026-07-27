#lang racket
(define (right_angle_triangle a b c)
  (define s (sort (list a b c) <))
  (= (+ (* (first s) (first s)) (* (second s) (second s)))
     (* (third s) (third s))))
