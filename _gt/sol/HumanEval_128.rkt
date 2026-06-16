#lang racket
(define (prod_signs arr)
  (if (null? arr) #f
      (* (apply + (map abs arr)) (apply * (map sgn arr)))))
