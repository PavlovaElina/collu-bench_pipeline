#lang racket
(define (mean_absolute_deviation numbers)
  (define m (/ (apply + numbers) (length numbers)))
  (/ (apply + (map (lambda (x) (abs (- x m))) numbers)) (length numbers)))
